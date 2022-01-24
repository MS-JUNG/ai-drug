from pychem.pychem import PyChem2d, PyChem3d
import pandas as pd 
from rdkit import Chem , DataStructs
from rdkit.Chem import AllChem
import string
from pychem import bcut
from sklearn.externals import joblib
import numpy as np

from rdkit.Chem import MACCSkeys
# smiles = pd.read_csv('smiles.csv').iloc[:, 0]
# alldes = {}
# smiles = list(smiles)

# Chem.MolFromSmiles



class reg():
    # n = 1 : ecfp2, n = 2 : ecfp4, n = 3 : ecfp6
    def __init__(self,smiles, n):
        # super()._init__()
        

        self.smiles = list(pd.read_csv(smiles).iloc[:,0])
        self.mols = []
        self.n = n 
        

        for i in range(len(self.smiles)):
            self.mols.append(Chem.MolFromSmiles(self.smiles[i]))
        # self.mol = Chem.MolFromSmiles(string.strip(self.smile))
      
       



 

    def smile2descriptor(self):

    # descriptor order 
            Clreg = ['nsulph', 'VSAEstate8', 'QNmin', 'IDET', 'ndb', 'slogPVSA2', 'MATSv5', 'S32', 'QCss', 'bcutm4', 'S9', 'bcutp8', 'Tnc',
                 'nsb', 'Geto', 'bcutp11', 'S7', 'MATSm2', 'GMTIV', 'nhet', 'MATSe1', 'CIC0', 'bcutp3', 'Gravto', 'EstateVSA9', 
                'MATSe3', 'MATSe5', 'UI', 'S53', 'J', 'bcute1', 'MRVSA9', 'PEOEVSA0', 'MATSv2', 'IDE', 'AWeight', 'IC0', 'S16', 'bcutp1'
                     , 'PEOEVSA12']
            cacoreg = ['ncarb', 'IC0', 'bcutp1', 'bcutv10', 'GMTIV', 'nsulph', 'CIC6', 'bcutm12', 'S34', 'bcutp8', 'slogPVSA2', 'QNmin', 'LogP2', 'bcutm1', 'EstateVSA9',
             'slogPVSA1', 'Hatov', 'J', 'AW', 'S7', 'dchi0', 'MRVSA1', 'LogP',  'Tpc', 'PEOEVSA0', 'Tnc', 'S13', 'TPSA', 'QHss', 'ndonr']         

            Herg = ['ndb', 'nsb', 'ncarb', 'nsulph', 'naro', 'ndonr', 'nhev', 'naccr', 'nta', 'nring', 'PC6', 'GMTIV', 'AW', 
                    'Geto', 'BertzCT', 'J', 'MZM2', 'phi', 'kappa2', 'MATSv1', 'MATSv5', 'MATSe4', 'MATSe5', 'MATSe6', 'TPSA', 'Hy', 'LogP', 'LogP2', 'UI', 'QOss', 'SPP', 'LDI', 'Qass', 'QOmin', 'QNmax', 'Qmin', 'Mnc', 'EstateVSA7', 'EstateVSA0', 'EstateVSA3', 'PEOEVSA0', 'PEOEVSA6', 'MRVSA5', 'MRVSA4', 'MRVSA3', 'MRVSA6', 'slogPVSA1'] 




           
            CL_all = []
            Caco_all = []
            Herg_all = []
            maccs = []
            ecfp = []
        
            for i in range(len(self.smiles)):
    #smiles2maccs
                temp_maccs = list(MACCSkeys.GenMACCSKeys(self.mols[i]).ToBitString())
                # temp_maccs.insert(0, self.smiles[i])
                
                maccs.append(temp_maccs)

    #smiles2ecfp             
                obj = AllChem.GetMorganFingerprintAsBitVect(self.mols[i], radius = self.n,  nBits=2048, useChirality=True)
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(obj,arr)

                temp_ecfp = list(arr)
                # temp_ecfp.insert(0, self.smiles[i])
                ecfp.append(temp_ecfp)




                drugclass =PyChem2d()
                drugclass.ReadMolFromSmile(self.smiles[i])
                Const =  drugclass.GetConstitution()
                Charge = drugclass.GetCharge()
                Kappa = drugclass.GetKappa()
                Moe = drugclass.GetMOE()
                Topo = drugclass.GetTopology()
                Estate = drugclass.GetEstate()
                Moran = drugclass.GetMoran()
                Basak = drugclass.GetBasak()
                Mlp  = drugclass.GetMolProperty()
                Connectivity = drugclass.GetConnectivity()
                Burden = drugclass.GetBcut()

                all = [Const,Charge, Moe, Topo, Estate, Moran, Basak, Mlp, Burden, Connectivity,Kappa]

                ALL = {}
                for i in all:
                
                     ALL.update(i)
                temp_Cl = []
                
                temp_Caco = []
                temp_herg = []
                # temp_Caco.insert(0,self.smiles[i])
                # temp_Cl.insert(0,self.smiles[i])
    #smiles2CL input descriptor
                for i in Clreg:
                    temp_Cl.append(ALL[i])
            
    #smils2Caco input descriptor
                for i in cacoreg:
                    temp_Caco.append(ALL[i])
     #smils2Herg input descriptor
                for i in Herg:
                    temp_herg.append(ALL[i])

               
            
            
                CL_all.append(temp_Cl)
                Caco_all.append(temp_Caco)
                Herg_all.append(temp_herg)
        
            

  
            return CL_all, Caco_all, maccs, ecfp, Herg_all


    
    


                

def predict_all(smiles):
    # n = 1 : ecfp2, n = 2 : ecfp4, n = 3 : ecfp6

    data = reg(smiles,2)
    
    cl = data.smile2descriptor()[0]
    caco = data.smile2descriptor()[1]
    
    maccs = data.smile2descriptor()[2]
    ecfp = data.smile2descriptor()[3]
    herg = data.smile2descriptor()[4]

    # maccs = np.array(maccs)
    # ecfp = np.array(ecfp)
    # ecfp_del = np.delete(ecfp,0,axis=1)

    # cl = np.array(cl)
    # # cl_del = np.delete(cl,0,axis =1 )
    # caco = np.array(caco)
    # herg = np.array(herg)
    # caco_del = np.delete(caco,0,axis =1 )


    # model load 
    CL_regression = joblib.load('C:/Users/pc/Desktop/admet_predictor/regression_model/CL/CL_Model.pkl')
    Caco_regresion = joblib.load('C:/Users/pc/Desktop/admet_predictor/regression_model/caco2/caco2_Model.pkl')
    CYP3A4 =  joblib.load('C:/Users/pc/Desktop/admet_predictor/classification_model/CYP3A4-inhibitor/CYP_inhibitor_3A4_SVC_ecfp4_model.pkl')
    hia = joblib.load('C:/Users/pc/Desktop/admet_predictor/classification_model/HIA/model_0.pkl')            
    BBB = joblib.load('C:/Users/pc/Desktop/admet_predictor/classification_model/BBB/BBB_RF_ecfp2_model.pkl')
    Pgp = joblib.load('C:/Users/pc/Desktop/admet_predictor/classification_model/Pgp-inhibitor/PGP_inhibitor_SVC_ecfp4_model.pkl')
    Herg = joblib.load('C:/Users/pc/Desktop/admet_predictor/classification_model/heRG/herg_model.pkl')
    return hia.predict(maccs),Caco_regresion.predict(caco), BBB.predict(ecfp), Pgp.predict(ecfp),CYP3A4.predict(ecfp),CL_regression.predict(cl),Herg.predict(herg)


if __name__ == "__main__":


    print(predict_all('C:/Users/pc/Desktop/ADMET_final/smiles.csv'))


# print(joblib.load('C:/Users/pc/Desktop/admet_predictor/regression_model/LABEL/caco2_Label.pkl'))

