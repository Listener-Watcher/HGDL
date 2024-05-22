import matplotlib.pyplot as plt


import numpy as np



# #fig,ax = plt.subplots()
# kl = np.array([0.8241,0.8272,0.8588,0.8927,0.9612,1.1466,1.4362,1.8297,2.6431,4.0711])
# kl2 = np.array([0.4321,0.381,0.6133,0.7007,0.8621,0.9637,1.5131,2.1378,4.1235,4.6785])
# x = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
# plt.plot(x,kl,label="ACM")
# plt.plot(x,kl2,label="DRUG")
# plt.xlabel("edge drop rate",fontsize=12)
# plt.ylabel("KL loss",fontsize=12)
# plt.grid()
# plt.legend(fontsize=10)
# plt.show()

drug_1 = np.array([0.4026,np.log(11.0937),0.3266,np.log(2.3046),0.5524,0.8677])
drug_2 = np.array([0.1756,np.log(8.9557),0.2275,np.log(1.9152),0.6985,0.3857])
drug_3 = np.array([0.3946,np.log(11.1627),0.3232,np.log(2.3233),0.5535,0.8599])
drug_hgdl = np.array([0.168,np.log(9.0454),0.2241,np.log(1.9121),0.7009,0.3955])

acm_1 = np.array([0.2208,np.log(13.1462),0.3657,np.log(3.45),0.6241,0.8786])
acm_2 = np.array([0.5546,np.log(13.486),0.7012,np.log(3.5959),0.2385,1.9286])
acm_3 = np.array([0.4198,np.log(13.2984),0.6013,np.log(3.4953),0.3639,1.392])
acm_hgdl = np.array([0.2014,np.log(13.0967),0.3533,np.log(3.3477),0.6364,0.7739])

dblp_1 = np.array([0.0166,np.log(2.789),0.0539,np.log(1.6314),0.9457,0.0498])
dblp_2 = np.array([0.3338,np.log(3.1914),0.5551,np.log(1.694),0.4275,0.9615])
dblp_hgdl = np.array([0.0166,np.log(2.7876),0.0525,np.log(1.6311),0.9472,0.0469])

yelp_1 = np.array([0.3385,np.log(7.3524),0.4799,np.log(2.6023),0.4432,1.0657])
yelp_2 = np.array([0.3391,np.log(7.3537),0.4803,np.log(2.6022),0.4425,1.0668])
yelp_hgdl = np.array([0.3383,np.log(7.3546),0.4778,np.log(2.6027),0.4458,1.065])

urban_1 = np.array([0.5224,np.log(8.373),0.556,np.log(2.7866),0.3145,1.4844])
urban_2 = np.array([0.5246,np.log(8.3769),0.5585,np.log(2.7858),0.3074,1.478])
urban_3 = np.array([0.5185,np.log(8.3658),0.5556,np.log(2.785),0.3155,1.4716])
urban_hgdl = np.array([0.4986,np.log(8.3751),0.5352,np.log(2.7866),0.3404,1.4122])

barWidth = 0.2
fig = plt.subplots(figsize=(12,8))
ft = 20
plt.rc('font', size=ft) # This line makes it work!
br1 = np.arange(dblp_1.shape[0])
br2 = [x+barWidth for x in br1]
br3 = [x+barWidth for x in br2]
br4 = [x+barWidth for x in br3]
#urbru
plt.bar(br1,acm_1,color='b',width=barWidth,edgecolor='grey',label=r"$p_{1}=apa$",)
plt.bar(br2,acm_2,color='y',width=barWidth,edgecolor='grey',label=r'$p_{2}=afa$')
plt.bar(br3,acm_3,color='g',width=barWidth,edgecolor='grey',label=r'$p_{3}=apspa$')
plt.bar(br4,acm_hgdl,color='r',width=barWidth,edgecolor='grey',label='HGDL')
# plt.xlabel(fontsize=ft)
plt.ylabel("Measure",fontsize=ft)
plt.grid()
plt.xticks([r+barWidth for r in range(dblp_1.shape[0])],['COD','CAD','CHD','CLD','IND','KL'],fontsize=ft)
plt.legend(fontsize=ft,loc='lower center',bbox_to_anchor=(0.5, -0.17),
          ncol=5, fancybox=False, shadow=False)
plt.show()