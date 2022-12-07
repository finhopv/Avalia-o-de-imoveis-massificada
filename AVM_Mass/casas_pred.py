import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unidecode import unidecode
import janitor as jn
import joblib
import time
import re
from urllib import response
from flask import Flask, render_template, request, session, make_response, send_file
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from unidecode import unidecode

#*** Flask configuration

 
# Define folder to save uploaded files to process further
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
 
# Define allowed files (for this example I want only csv file)
ALLOWED_EXTENSIONS = {'csv'}
 
app = Flask(__name__, template_folder= "templates", static_folder='staticFiles')
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'
 

def get_good_cep(x):
    x = re.sub("[^0-9]","",x)
    if len(x) > 8:
        x = x[:8]
    while len(x) < 8:
        x = "0" + x
    return x

def treat_txt(x):
    x = unidecode(x).upper()
    x = re.sub("[^A-Z0-9]"," ",x)
    x = re.sub("\s+", " ", x).strip()
    return x

class Preditor(object):
    def __init__(self, model_path='model/RFR_d_8regional_house_v6-Original.model',Error_path='model/DT_classificação_casas.model'):
        self.modelo = joblib.load(model_path)
        self.error_class=joblib.load(Error_path)
        
        df_macro = pd.read_csv("datasets/Indices_Mun.csv", encoding='ISO-8859-1')
        df_macro = jn.clean_names(df_macro)
        df_macro['idh'] = pd.to_numeric(df_macro['idh'].map(lambda item: str(item).replace(',', '.').replace('-', '0')),
                                downcast='float')
        df_macro['municipios'] = df_macro['municipios'].map(lambda item: str(item).strip().upper())
        df_macro['pib_per_capita'] = pd.to_numeric(df_macro['pib_per_capita'].map(lambda item: str(item).strip().replace(',', '.')),
                                downcast='float')
        df_macro['pib_anual'] = pd.to_numeric(df_macro['produto_interno_bruto_anual_r$_1_000_'].map(lambda item: str(item).strip().replace(',', '.')),
                                downcast='float')
        self.macro = df_macro[["uf","municipios",'pib_per_capita',"idh","pib_anual"]].set_index(["uf","municipios"])
    #----------------------------------------------------------------------------
    # converte o dicionário no formato do dataframe esperado pelo modelo
    def trata_dados(self,dicionario):
        if "translator" in self.modelo.keys():
            keys=dicionario.keys()
            for k in keys:
                dicionario[self.modelo['translator'][k]]=dicionario.pop(k)
        ##lista de todos os campos utilizados
        camp_util = self.modelo["input_features"]
        # valida campos texto do cobol
        for c in camp_util.keys():
            try:
                teste = dicionario[c]
            except KeyError:
                raise Exception(1, f"Coluna {c} no tiene valor")
                #dicionario[c] = ""
        ## batch or online
        try:
            len(dicionario[c])
            if type(dicionario[c])!=str:
                val_is_list = True
            else:
                val_is_list = False
        except:
            val_is_list = False
        ## load data
        try:
            if val_is_list:
                ddf = pd.DataFrame.from_dict(dicionario)
            else:
                ddf = pd.DataFrame.from_dict([dicionario])
        except:
            raise Exception(1, "Valores de entrada no son validos")
        ## check data types
        for k,v in camp_util.items():
            try:
                ddf[k] = ddf[k].astype(v)
            except:
                raise Exception(1, f"Valor {k} nao e um objeto do tipo {v}")
        ## fix data
        reg_cols = ['tx_cid_imv','sg_uf_imv','tx_bai_imv']
        for c in reg_cols:
            ddf[c] = ddf[c].apply(lambda x: treat_txt(x))
        ddf["CEP"] = ddf["CEP"].apply(lambda x: get_good_cep(x))
        ddf['cep_div_subsetor'] = np.nan
        ddf['cep_div_subsetor'] = ddf['CEP'].str.slice(start=0,stop=5)
        ddf['cep_subsetor']     = ddf['CEP'].str.slice(start=0,stop=4)
        ddf['cep_setor']        = ddf['CEP'].str.slice(start=0,stop=3)
        ddf['cep_subregiao']    = ddf['CEP'].str.slice(start=0,stop=2)
        ddf['cep_regiao']       = ddf['CEP'].str.slice(start=0,stop=1)
        ## add macro economic data
        ddf = ddf.merge(self.macro
                        , how="left"
                        , left_on=['tx_cid_imv','sg_uf_imv']
                        , right_on=["municipios","uf"] )
        ddf.fillna({"idh":0, 'pib_anual':0, 'pib_per_capita':0}, inplace=True)
        ## add regional data
        all_ucb_dfs = self.modelo["regional_data"]
        ddf['vl_avlc_hist_regional'] = np.nan
        for f, mf, f_df in all_ucb_dfs:
            ddf = ddf.reset_index().merge(f_df, on=mf, how="left", suffixes=(None,"_"+f)).set_index("index")
            ddf['vl_avlc_hist_regional'].fillna(ddf['vl_avlc_hist_regional_'+f], inplace=True)
            ddf['vl_avlc_hist_regional_'+f].fillna(0, inplace=True)
        ## add cep data
        all_cep_dfs = self.modelo["cep_model"]
        ddf['vl_cep_estr_hist'] = np.nan
        for f, f_df in all_cep_dfs:
            ddf = ddf.reset_index().merge(f_df, on=f, how="left", suffixes=(None,"_"+f)).set_index("index")
            ddf['vl_cep_estr_hist'].fillna(ddf['vl_cep_estr_hist_'+f], inplace=True)
            ddf['vl_cep_estr_hist_'+f].fillna(0, inplace=True)
        # retorna o Dataframe que o modelo espera
        return ddf[self.modelo["feature_columns"]]
    #----------------------------------------------------------------------------
    # Realiza a previsão
    def executar(self, dic):

        #get_vars()
        
        if type(dic) != dict:
            raise Exception(0,"Dado recebido nao e um dicionario")
        # formata o dicionario para o data frame esperado
        try:
            X = self.trata_dados(dic)
        except Exception as ex:
            raise Exception(2,ex)
        # reforna a valor estimado do metro cuadrado do imovel
        predicao_m2 = self.modelo["model"].predict(X)[0]
        predicao = predicao_m2*float(dic['vl_area_ttl_pvt'])
        
        X['vl_train_set']=0
        X['uf_cid_bai']=int(X['vl_avlc_hist_regional_bai']==0)+int(X['vl_avlc_hist_regional_cid']==0)+int(X['vl_avlc_hist_regional_uf']==0)
        X['hist_cep']=int(X['vl_cep_estr_hist_cep_div_subsetor']==0)+int(X['vl_cep_estr_hist_cep_subsetor']==0)+\
        int(X['vl_cep_estr_hist_cep_setor']==0)+int(X['vl_cep_estr_hist_cep_subregiao']==0)+int(X['vl_cep_estr_hist_cep_regiao']==0)

        classe_erro=self.error_class['model'].predict_proba(X[self.error_class['input_features']])[:,0]
        if classe_erro>0.8 and (X['uf_cid_bai'].values<1 and X['hist_cep'].values<2):
            classe_erro='Bom'
        elif classe_erro<0.5:
            classe_erro='Incerto'
        else:
            classe_erro='Regular'
            
            
        return predicao #{'valorAvaliadoImovel': predicao,
                #'textoCategoriaErro':classe_erro,
               #'timestampPesquisa': time.time()}
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


        #base.to_excel('bas.xlsx') ##
        #base=pd.read_excel('bas.xlsx') ##

        start_time = time.time()
        preditor = Preditor()
        print('model load: %s segundos' % (time.time() - start_time))

        predit=[]
        for i in range(base.shape[0]):
            
          idade_imv=base['numeroIdadeImovel'][i]#
          vl_tt_ter=base['valorAreaTotalTerreno'][i]#
          qto_ban=base['quantidadeQuartosComBanheiros'][i]#
          cod_Pisc=base['codigoInfraestruturaPiscinaPredio'][i]#
          vl_tt_pvt=base['valorAreaTotalPrivativo'][i]#
          nm_vag_cb=base['NumeroVagaCoberta'][i]#
          cod_pdr=base['codigoPadraoAperfeicoamentoImovel'][i]#
            # valor avaliacao historico regional
          txt_cep=base['textoCepImovel'][i]#
          sigla_UF=base['siglaUnidadeFederativaImovel'][i]#
          cidade=base['textoCidadeImovel'][i]#
          bairro=base['textoBairroImovel'][i]#

        info = {'codigoInfraestruturaPiscinaPredio': cod_Pisc,
             'codigoPadraoAperfeicoamentoImovel': cod_pdr,
             'numeroIdadeImovel': idade_imv,
             'numeroVagaCobertura': nm_vag_cb,
             'quantidadeQuartoComBanheiro': qto_ban,
             'valorAreaTotalPrivativo': vl_tt_pvt,
             'valorAreaTotalTerreno':vl_tt_ter,
             'siglaUnidadeFederacaoImovel': sigla_UF,
             'textoCidadeImovel': cidade,
             'textoBairroImovel': bairro,
             'textoCepImovel': txt_cep
             }
          
          
          #teste = np.array([[sexo,casado,dependentes,educacao,trabalho_conta_propria,rendimento,valoremprestimo]])
        predit.append(preditor.executar(info))
    base['previsao']=predit	
    base.to_csv('base_prevista.csv', index=False)  
        # Storing uploaded file path in flask session
    session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
 
    return render_template('front.html')


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
#if __name__ == '__main__':
#    import time

#    start_time = time.time()
#    preditor = Preditor(model_path='./data/production/RFR_d_8regional_house_v6.model',Error_path='./data/production/DT_classificação_casas.model')
#    print('model load: %s segundos' % (time.time() - start_time))  
    
 #   info = {'codigoInfraestruturaPiscinaPredio': 0,
 #           'codigoPadraoAperfeicoamentoImovel': 0.0,
 #           'numeroIdadeImovel': 7.0,
 #           'numeroVagaCobertura': 1.0,
 #           'quantidadeQuartoComBanheiro': 1.0,
 #           'valorAreaTotalPrivativo': 75.98,
 #           'valorAreaTotalTerreno':100,
 #           'siglaUnidadeFederacaoImovel': 'GO',
 #           'textoCidadeImovel': 'GOIÂNIA',
 #           'textoBairroImovel': 'PARQUE AMAZONIA',
 #           'textoCepImovel': '74843200'}  
 #       
 #   start_time = time.time()
 #   print(preditor.executar(info))
 #   print('prediction %s segundos' % (time.time() - start_time))
