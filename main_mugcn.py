import torch
from sklearn.metrics import f1_score
#from utils import EarlyStopping, load_data
from scipy.spatial import distance
from utils_ import clark,intersection
from utils_mugcn_pre import *
from model_mugcn import HAN_MUGCN
eps = 1e-9
def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    _, indices2 = torch.max(labels,dim=1)
    prediction = indices.long().cpu().numpy()
    #labels = labels.cpu().numpy()
    labels_ = indices.long().cpu().numpy()

    accuracy = (prediction == labels_).sum() / len(prediction)
    micro_f1 = f1_score(labels_, prediction, average="micro")
    macro_f1 = f1_score(labels_, prediction, average="macro")
    score_cosin = 0
    score_can = 0
    score_cheb = 0
    for i in range(labels.shape[0]):
        score_cosin += distance.cosine(labels[i].cpu().detach().numpy(),logits[i].cpu().detach().numpy())
        score_can += distance.canberra(labels[i].cpu().detach().numpy(),logits[i].cpu().detach().numpy())
        score_cheb += distance.chebyshev(labels[i].cpu().detach().numpy(),logits[i].cpu().detach().numpy())
    score_cosin/= labels.shape[0]
    score_can/=labels.shape[0]
    score_cheb/=labels.shape[0]
    score_clark = clark(logits+eps,labels+eps)
    score_intersection = intersection(logits,labels)
    print(f'cosin distance:{score_cosin:.4f}')
    print(f'canberra distance:{score_can:.4f}')
    print(f'chebyshev distance:{score_cheb:.4f}')
    print(f'clark distance:{score_clark:.4f}')
    print(f'intersection distance:{score_intersection:.4f}')
    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func((logits[mask]+1e-9).log(), labels[mask]+1e-9)
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1


def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    (
        g,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
    ) = load_data(args.dataset)

    if hasattr(torch, "BoolTensor"):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = features.to(args.device)
    labels = labels.to(args.device)
    train_mask = train_mask.to(args.device)
    val_mask = val_mask.to(args.device)
    test_mask = test_mask.to(args.device)
    g = g.to(args.device)
    model = HAN_MUGCN(
        g,
        #meta_paths=[[("author","to","paper"),("paper","to","author")], [("author","to","paper"),("paper","to","term"),("term","to","paper"),("paper","to","author")],[("author","to","paper"),("paper","to","conference"),("conference","to","paper"),("paper","to","author")]],
        #meta_paths=[[("author","to","paper"),("paper","to","term"),("term","to","paper"),("paper","to","author")],[("author","to","paper"),("paper","to","conference"),("conference","to","paper"),("paper","to","author")]], 
        #meta_paths=[[("author","to","paper"),("paper","to","author")],[("author","to","paper"),("paper","to","conference"),("conference","to","paper"),("paper","to","author")]],  
        meta_paths=[[("author","to","paper"),("paper","to","conference"),("conference","to","paper"),("paper","to","author")]],
        #meta_paths=[[("author","to","paper"),("paper","to","term"),("term","to","paper"),("paper","to","author")]],
        #meta_paths =[[("author","to","paper"),("paper","to","author")]],
        hidden_size=args.gcnhidden,
        out_size=num_classes,
        device = args.device,
        train_mask = train_mask,
        val_mask = val_mask,
        test_mask = test_mask,
        args = args,
    ).to(args.device)
    #g = g.to(args["device"])
    stopper = EarlyStopping(patience=args.patience)
    #loss_fcn = torch.nn.CrossEntropyLoss()
    loss_fcn = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.005, weight_decay=0
    )

    for epoch in range(10000):
        model.train()
        logits = model(g, features)
        loss = loss_fcn((logits[train_mask]+1e-9).log(), labels[train_mask]+1e-9)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(
            logits[train_mask], labels[train_mask]
        )
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(
            model, g, features, labels, val_mask, loss_fcn
        )
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print(
            "Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | "
            "Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}".format(
                epoch + 1,
                loss.item(),
                train_micro_f1,
                train_macro_f1,
                val_loss.item(),
                val_micro_f1,
                val_macro_f1,
            )
        )

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(
        model, g, features, labels, test_mask, loss_fcn
    )
    print(
        "Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(
            test_loss.item(), test_micro_f1, test_macro_f1
        )
    )


if __name__ == "__main__":
    import argparse
    #from utils import setup
    parser = argparse.ArgumentParser("HAN")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    """parser.add_argument(
        "-ld",
        "--log-dir",
        type=str,
        default="results",
        help="Dir for saving training results",
    )"""
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='gpu or cpu')
    parser.add_argument('--epochs', type=int, default=6000)
    parser.add_argument('--dataset',type=str,default='dblp')
    parser.add_argument('--gcnhidden', type=int, default=300)
    parser.add_argument('--pgehidden',type=int,default=64)
    parser.add_argument('--decoderhidden',type=int,default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--alpha',type=float,default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--gcnllayers',type=int,default=2)
    parser.add_argument('--gcnvlayers',type=int,default=3)
    parser.add_argument('--pgelayers',type=int,default=2)
    parser.add_argument('--freqv',type=int,default=100)
    parser.add_argument('--freql',type=int,default=100)
    parser.add_argument('--patience',type=int,default=140)
    parser.add_argument('--decoderlayers',type=int,default=3)
    parser.add_argument('--mode',type=str,default='dynamic',help='dynamic,static')
    parser.add_argument('--normalize',action='store_true')
    parser.add_argument('--inner_epochs',type=int,default=70)
    parser.add_argument('--p',type=int,default=50)


    #args = parser.parse_args().__dict__
    args = parser.parse_args()
    args = setup(args)

    main(args)
