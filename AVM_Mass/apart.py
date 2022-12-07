from urllib import response
from flask import Flask, render_template, request, session, make_response, send_file
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
import joblib
import time
from datetime import datetime
from unidecode import unidecode
import janitor as jn
import re

#*** Flask configuration

 
# Define folder to save uploaded files to process further
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
 
# Define allowed files (for this example I want only csv file)
ALLOWED_EXTENSIONS = {'csv'}
 
app = Flask(__name__, template_folder= "templates", static_folder='staticFiles')
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
#model = joblib.load('model/model.pkl')

#base=pd.read_csv('datasets/emprestimo.csv')

# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'
 
## Class Preditor
class Preditor(object):

    def __init__(self):
        #self.modelo = pickle.load('Modelo/RL.pickle')
        #df_macro = pd.read_csv("Modelo/Indices_Mun.csv", encoding='ISO-8859-1')

        self.modelo = joblib.load('model/RFR_d_8regional_apartment_v6.model')
        df_macro = pd.read_csv("datasets/Indices_Mun.csv", encoding='ISO-8859-1')

        df_macro = jn.clean_names(df_macro)
        df_macro['pib_anual'] = pd.to_numeric(df_macro['produto_interno_bruto_anual_r$_1_000_'].map(lambda item: str(item).strip().replace(',', '.')),
                                downcast='float')
        df_macro['municipios'] = df_macro['municipios'].map(lambda item: str(item).strip().upper())
        self.macro = df_macro[["uf","municipios","pib_anual"]].set_index(["uf","municipios"])
    #----------------------------------------------------------------------------
    # converte o dicionário no formato do dataframe esperado pelo modelo
    def trata_dados(self,dicionario):

        ##lista de todos os campos utilizados
        camp_util = {
            'numeroIdadeImovel':int#'nr_idd_imv'
            , 'numeroUnidadesPorAndar':int#'nr_und_adar'
            , 'quantidadeQuartosComBanheiros':int#'qt_qrto_com_bnro'
            # valor avaliacao historico regional
            , 'siglaUnidadeFederativaImovel':str#'sg_uf_imv'
            , 'textoCidadeImovel':str#'tx_cid_imv'
            , 'textoBairroImovel':str#'tx_bai_imv'
            # valor avaliacao historico cep and 2km
            , 'valorLongitudeLocalizacaoImovel':float#'vl_lgte_lczc_imv'
            , 'valorLatitudeLocalizacaoImovel':float#'vl_ltd_lczc_imv'
            # qtd_ietr_pred
            , 'codigoInfraestruturaPiscinaPredio':int#'cd_ietr_psca_ped'
            , 'codigoInfraestruturaQuadraPredio':int#'cd_ietr_qdra_ped'
            , 'codigoInfraestruturaLazerPredio':int#'cd_ietr_lz_ped'
            , 'codigoInfraestruturaEspacoAlimentacao':int#'cd_ietr_esp_alm'
            , 'codigoInfraestruturaSalaEventos':int#'cd_ietr_sala_evt'
            # vl_area_ttl_pvt
            , 'valorAreaTotalTerreno':float#'vl_area_ttl_trrn'
            , 'valorAreaTerrenoComum':float#'vl_area_trrn_cmum'
        }
        # valida campos texto do cobol
        for c in camp_util.keys():
            try:
                teste = dicionario[c]
            except KeyError:
                raise Exception(1, f"Coluna {c} no tiene valor")
                #dicionario[c] = ""

        data_dict = dict()
        for col, tipo in camp_util.items():
            data_dict[col] = self.check_type(dicionario, col, tipo)

        # get gps based data
        coords = np.array([[data_dict['valorLatitudeLocalizacaoImovel']
                            , data_dict['valorLongitudeLocalizacaoImovel']]])
        v0,v1,v2,v3 = self.get_neighbor_feat(coords)

        uf = data_dict['siglaUnidadeFederativaImovel']
        if len(uf)>2:
            raise Exception(1,"Sigla Unidade Federativa do imovel e invalida")
        cid = unidecode(data_dict['textoCidadeImovel']).upper()
        bai = unidecode(data_dict['textoBairroImovel']).upper()

        # get regional based data
        df_reg = self.modelo["regional_data"]
        if (uf, cid, bai) in df_reg.index:
            reg_val = df_reg.loc[(uf, cid, bai)]["vl_avlc_m2_imv_bai"].mean()
        elif (uf, cid) in df_reg.index:
            reg_val = df_reg.loc[(uf, cid)]["vl_avlc_m2_imv_cid"].mean()
        elif (uf) in df_reg.index:
            reg_val = df_reg.loc[(uf)]["vl_avlc_m2_imv_uf"].mean()
        else:
            raise Exception(1, f"No regional data available for {uf} {cid} {bai}")

        # get macro data
        if (uf, cid) in self.macro.index:
            pib_anual = self.macro.loc[(uf,cid)]["pib_anual"]
        else:
            raise Exception(1, f"No macro economic data available for {uf} {cid}")

        # sumar estos valores
        qtd_ietr_pred = [
            'codigoInfraestruturaPiscinaPredio'
            , 'codigoInfraestruturaQuadraPredio'
            , 'codigoInfraestruturaLazerPredio'
            , 'codigoInfraestruturaEspacoAlimentacao'
            , 'codigoInfraestruturaSalaEventos'
        ]
        # Convert o dicionario recebido no formato esperado
        cli  = [{
            'nr_idd_imv': data_dict['numeroIdadeImovel']
            , 'nr_und_adar': data_dict['numeroUnidadesPorAndar']
            , 'pib_anual': pib_anual
            , 'qt_qrto_com_bnro': data_dict['quantidadeQuartosComBanheiros']
            , 'vl_area_ttl_pvt': data_dict['valorAreaTotalTerreno'] - data_dict['valorAreaTerrenoComum']
            , 'qtd_ietr_pred': sum(data_dict[c] for c in qtd_ietr_pred)
            , 'vl_ltd_lczc_imv': data_dict['valorLatitudeLocalizacaoImovel']
            , 'vl_ltd_lczc_imv': data_dict['valorLatitudeLocalizacaoImovel']
            , 'vl_avlc_hist_regional_t': reg_val
            , 'vl_avlc_hist_cep_avg': v0[0]
            , 'vl_avlc_hist_2km_avg': v2[0]
        }]

        # retorna o Dataframe que o modelo espera
        return pd.DataFrame(cli)
    #----------------------------------------------------------------------------
    # check data types are as expected
    def check_type(self, dic, col, tipo):
        try:
            r = tipo(dic[col])
            return r
        except ValueError as ex:
            raise Exception(1, f"Valor {col} nao e um objeto do tipo {tipo.__name__}")
    #----------------------------------------------------------------------------
    # get gps calculated columns
    def get_neighbor_feat(self, co):
        n_neighbors=30
        maxdis=2.0
        #all computations exclude same building and values of current year(curr_yr=2020)
        vl_avkc_hist_local_v0 = np.zeros(len(co)) #average within same cep location (distance=0)
        vl_avkc_hist_local_v1 = np.zeros(len(co)) #number neighbors same cep location (distance=0)
        vl_avkc_hist_local_v2 = np.zeros(len(co)) #average outside same cep location but closer than maxdis=2km (0<distance<2)
        vl_avkc_hist_local_v3 = np.zeros(len(co)) #number of neighbors  (0<distance<2)
        i=0
        target_col = "vl_avlc_m2_imv"
        df_tmpy = self.modelo["cep_model"]["df_ref"][[target_col]]
        tree = self.modelo["cep_model"]["kdtree"]
        for coo in co: #we can do this loop in parallel if performance required.. or use dask!
            dist,ix = tree.query(coo,n_neighbors) #indices and distances SLOW
            nzero=n_neighbors - np.count_nonzero(dist) -1
            if (nzero > n_neighbors-6): #extend search if too many buildings same cep
                dist,ix = tree.query(coo,95)
                nzero=95 - np.count_nonzero(dist) -1
            dist=dist*105.0 # proxy for average degree transformation to km for location at -30S lat. (error estimate 3% max)
            nzero2=np.count_nonzero(dist < maxdis) -nzero #number distances below maxdis threshold and >0
            #print(i,nzero,nzero2,ix[0])
            mean0=0.
            mean2=0.
            nzerof=nzero
            nzero2f=nzero2
            try:
                df_tmpx = df_tmpy.iloc[ix] #get data of neighbors SLOW!!
            except:
                print("index error:",ix)
                print("df_shape:",df_tmpy.shape)
                raise
            if nzero>0:
                df_tmp = df_tmpx[0:nzero]
                nzerof=0
                if len(df_tmp)>0:
                    res=df_tmp[target_col].describe().to_numpy() #count,mean,std,min,25,50,75,max
                    mean0=res[1]
                    nzerof=res[5] #std
            if nzero2>0:
                df_tmp = df_tmpx[nzero+1:nzero+nzero2]
                nzero2f=0
                if len(df_tmp)>0:
                    res=df_tmp[target_col].describe().to_numpy() #count,mean,std,min,25,50,75,max
                    mean2=res[1]
                    nzero2f=res[5]
            if mean0==0: #possible value to fill undefined?
                mean0=df_tmpx[target_col].head(5).mean() #2
            if mean2==0: #possible value to fill undefined?
                mean2=df_tmpx[target_col].head(5).mean()
            if nzerof==0: #possible value to fill undefined?
                nzerof=df_tmpx[target_col].head(5).median()
            if nzero2f==0: #possible value to fill undefined?
                nzero2f=df_tmpx[target_col].head(5).median()
            vl_avkc_hist_local_v0[i]=mean0
            vl_avkc_hist_local_v1[i]=nzerof #std
            vl_avkc_hist_local_v2[i]=mean2
            vl_avkc_hist_local_v3[i]=nzero2f
            i=i+1
        #print('complete!')
        return vl_avkc_hist_local_v0,vl_avkc_hist_local_v1,vl_avkc_hist_local_v2,vl_avkc_hist_local_v3
    #----------------------------------------------------------------------------
    # Realiza a previsão
    def executar(self, dic):

        if type(dic) != dict:
            raise Exception(0,"Dado recebido nao e um dicionario")

        # formata o dicionario para o data frame esperado
        try:
            X = self.trata_dados(dic)
        except Exception as ex:
            raise Exception(2,ex)

        # reforna a valor estimado do metro cuadrado do imovel
        predicao = self.modelo["model"].predict(X)[0]
        
        return ( predicao)     
        #return {'valorEstimadoAvaliacaoM2Imovel': predicao}


def executor(dicionario):
    preditor = Preditor()
    x = preditor.executar(info)
    return x
@app.route('/')
def index():
    return render_template('front.html')
 
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # upload file flask
        uploaded_df = request.files['uploaded-file']
        
        # Extracting uploaded data file name
        data_filename = secure_filename(uploaded_df.filename)
 
        # flask upload file to database (defined uploaded folder in static path)
        uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
        base=pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], data_filename)) 

        ## tratar valores para o dicionário
        def trat(coorde):
            coorde=re.sub('[^0-9]', '', coorde)
            return coorde

        def neg(coorde):
            coorde= coorde * (-1)
            return coorde 

        base.valorLongitudeLocalizacaoImovel=base.valorLongitudeLocalizacaoImovel.apply(trat)
        base.valorLatitudeLocalizacaoImovel=base.valorLatitudeLocalizacaoImovel.apply(trat)

        base.valorLongitudeLocalizacaoImovel=base.valorLongitudeLocalizacaoImovel.astype(float)
        base.valorLatitudeLocalizacaoImovel=base.valorLatitudeLocalizacaoImovel.astype(float)

        base.valorLongitudeLocalizacaoImovel=base.valorLongitudeLocalizacaoImovel.apply(neg)
        base.valorLatitudeLocalizacaoImovel=base.valorLatitudeLocalizacaoImovel.apply(neg)

        #base.to_excel('bas.xlsx') ##
        #base=pd.read_excel('bas.xlsx') ##

        start_time = time.time()
        preditor = Preditor()
        print('model load: %s segundos' % (time.time() - start_time))

        predit=[]
        for i in range(base.shape[0]):
            
          idade_imv=base['numeroIdadeImovel'][i]
          unid_por_andar=base['numeroUnidadesPorAndar'][i]
          qto_ban=base['quantidadeQuartosComBanheiros'][i]
            # valor avaliacao historico regional
          sigla_UF=base['siglaUnidadeFederativaImovel'][i]
          cidade=base['textoCidadeImovel'][i]
          bairro=base['textoBairroImovel'][i]
            # valor avaliacao historico cep and 2km
          longitude=base['valorLongitudeLocalizacaoImovel'][i]
          print(type(longitude))
          latitude=base['valorLatitudeLocalizacaoImovel'][i]
            # qtd_ietr_pred
          cod_Pisc=base['codigoInfraestruturaPiscinaPredio'][i]
          cod_qda=base['codigoInfraestruturaQuadraPredio'][i]
          cod_lazer=base['codigoInfraestruturaLazerPredio'][i]
          cod_alim=base['codigoInfraestruturaEspacoAlimentacao'][i]
          cod_sala=base['codigoInfraestruturaSalaEventos'][i]
            # vl_area_ttl_pvt
          vl_ar_tot=base['valorAreaTotalTerreno'][i]
          vl_ar_cm=base['valorAreaTerrenoComum'][i]
          
          info = {
            'numeroIdadeImovel':idade_imv
            , 'numeroUnidadesPorAndar':unid_por_andar
            , 'quantidadeQuartosComBanheiros':qto_ban
            # valor avaliacao historico regional
            , 'siglaUnidadeFederativaImovel':sigla_UF
            , 'textoCidadeImovel':cidade
            , 'textoBairroImovel':bairro
            # valor avaliacao historico cep and 2km
            , 'valorLongitudeLocalizacaoImovel':longitude
            , 'valorLatitudeLocalizacaoImovel':latitude
            # qtd_ietr_pred
            , 'codigoInfraestruturaPiscinaPredio':cod_Pisc
            , 'codigoInfraestruturaQuadraPredio':cod_qda
            , 'codigoInfraestruturaLazerPredio':cod_lazer
            , 'codigoInfraestruturaEspacoAlimentacao':cod_alim
            , 'codigoInfraestruturaSalaEventos':cod_sala
            # vl_area_ttl_pvt
            , 'valorAreaTotalTerreno':vl_ar_tot
            , 'valorAreaTerrenoComum':vl_ar_cm
        }
          
          #teste = np.array([[sexo,casado,dependentes,educacao,trabalho_conta_propria,rendimento,valoremprestimo]])
          predit.append(preditor.executar(info))
    base['previsao']=predit	
    base.to_csv('base_prevista.csv', index=False)  
        # Storing uploaded file path in flask session
    session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
 
    return render_template('index_upload_and_show_data_page2.html')


@app.route('/show_data')
def showData():
    uploaded_df = pd.read_csv('base_prevista.csv')
 
    # pandas dataframe to html table flask
    uploaded_df_html = uploaded_df.to_html()
    index()
    return render_template('show_csv_data.html', data_var = uploaded_df_html)

 
    """ return send_file(
        'base_prevista.csv',
        mimetype='text/csv',
        download_name='AVM.csv',
        as_attachment=True
    ) """
     
 
if __name__=='__main__':
    app.run(debug = True)
