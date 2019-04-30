# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:32:20 2019

@author: jabm9
"""

import os
import re
from nltk.tokenize import word_tokenize
from pathlib import Path
from nltk.tokenize import WhitespaceTokenizer
import nltk.data
from nltk.tokenize import sent_tokenize
from nltk.parse.stanford import StanfordDependencyParser
from string import digits
import pickle




with open('./positive-words.txt', 'r') as file:
    texto_positivo = file.read()


with open('./negative-words.txt', 'r') as file:
    texto_negativo = file.read()
    
    
    
path_to_jar = './stanford-parser-full-2018-10-17/stanford-parser.jar'
path_to_models_jar = './stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar'

sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")



dependency_parser = StanfordDependencyParser(path_to_jar = path_to_jar,
path_to_models_jar = path_to_models_jar, 
model_path = 'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')


pathTrain = "./SFU_Review_Corpus_Raw_partitions/train/"
pathDev = "./SFU_Review_Corpus_Raw_partitions/dev/"
pathDev2 = "./SFU_Review_Corpus_Raw_partitions/dev2/"
pathDevPickle = "./SFU_Review_Corpus_Raw_partitions/devPickle/"
pathTest = "./SFU_Review_Corpus_Raw_partitions/test/"
corpus = ["BOOKS","CARS","COMPUTERS","COOKWARE","HOTELS","MOVIES","MUSIC","PHONES"]
corpus2 = ["BOOKS"]


palabras_positivas = word_tokenize(texto_positivo)
palabras_positivas = palabras_positivas[414:] #Elimina las 414 primeras entradas (corresponden a texto introductorio del documento)
palabras_negativas = word_tokenize(texto_negativo)
palabras_negativas = palabras_negativas[418:] #Elimina las 417 primeras entradas (corresponden a texto introductorio del documento)

#Vectores que contienen los pesos asignados a cada una de las palabras del lexicon
peso_palabras_positivas = [1] * 2006
peso_palabras_negativas = [1] * 4785

#Variables para conrolar el peso de adjetivos y verbos
peso_adjetivo = 5
peso_verbos = 2

patternAdjetivo= re.compile("^JJ")
patternVerbo = re.compile("VB")

#Variables para controlar el peso de las oraciones iniciales y finales
peso_oracion_inicial = 4
peso_oracion_final = 4

#Variable que define la cantidad a sumar a los pesos si aparece la palabra
valor_suma_pesos = 0.25

#Patrones para saber si los ficheros tienen una opinión positiva o negativa
#Los ficheros con yes en el nombre son opiniones positivas y los que tienen no son opiniones negativas
patternTextoPositivo= re.compile("^y")
patternTextoNegativo = re.compile("^n")

#Entrenamiento
numTextos = 0
numTextosPositivos = 0
numTextosNegativos = 0

whitespaceTokenizer = WhitespaceTokenizer()

#Metodo para realizar la actuación de pesos y encapsular codigo

def actualizarPesos(tokens_texto, palabras_lexicon, vector_pesos):
    valoresCompartidos = set(tokens_texto) & set(palabras_lexicon)
    for v in valoresCompartidos:
        indicePalabra = palabras_lexicon.index(v)
        vector_pesos[indicePalabra] += valor_suma_pesos

for c in corpus:
    pathTrainCorpus = pathTrain +"/"+c
    for fichero in os.listdir(pathTrainCorpus):
        with open(pathTrainCorpus+"/"+fichero,"r") as file:
            contents = Path(pathTrainCorpus+"/"+fichero).read_text()
            
        tokens_texto = whitespaceTokenizer.tokenize(contents)       #Tokenizar el texto
        [x.lower() for x in tokens_texto]                       #Transformar todas las palabras a minúscula para poder compararlas con el lexicón
        if patternTextoPositivo.match(fichero):
            actualizarPesos(tokens_texto,palabras_positivas,peso_palabras_positivas)
        else:
            actualizarPesos(tokens_texto,palabras_negativas,peso_palabras_negativas)


#Resultado: vectores de pesos actualizados en función del número de veces que aparezcan palabras positivas y negativas
#en el conjunto de entrenamiento


_POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
tagger = nltk.data.load(_POS_TAGGER)

sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")



listaResultados = []

#Preparación del sistema

def serializarDatosDev():
    for c in corpus:
        pathDevCorpus = pathDev + "/" +c
        for fichero in os.listdir(pathDevCorpus):
            with open(pathDevCorpus +"/"+fichero,"r") as file:
                contents = Path(pathDevCorpus +"/"+fichero).read_text()
                remove_digits = str.maketrans('', '', digits)
                contents = contents.translate(remove_digits)
            oracionesTexto = sentence_tokenizer.tokenize(contents)
            oracionesDependencias = []
            for oracion in oracionesTexto:
                iterator = dependency_parser.raw_parse(oracion)
                for dep in iterator:
                    lista = list(dep.triples())
                    oracionesDependencias.append(lista)
            print(pathDevCorpus +"/"+fichero)
            pickle.dump(oracionesDependencias, open(pathDevPickle+c+"/"+fichero,'wb'))    
            
def serializarDatosTest():
    for c in corpus:
        pathTestCorpus = pathTest + "/" +c
        for fichero in os.listdir(pathTestCorpus):
            with open(pathTestCorpus+"/"+fichero,"r") as file:
                contents = Path(pathTestCorpus+"/"+fichero).read_text()
                remove_digits = str.maketrans('', '', digits)
                contents = contents.translate(remove_digits)
            oracionesTexto = sentence_tokenizer.tokenize(contents)
            oracionesDependencias = []
            for oracion in oracionesTexto:
                iterator = dependency_parser.raw_parse(oracion)
                for dep in iterator:
                    lista = list(dep.triples())
                    oracionesDependencias.append(lista)
            print(pathTestCorpus+"/"+fichero)
            pickle.dump(oracionesDependencias, open(pathDevPickle+c+"/"+fichero,'wb'))   
            
#serializarDatosDev()
#serializarDatosTest()

#cargaDatos = pickle.load(open(pathDevPickle+corpus[0]+"/no9.txt",'rb'))


            




def evaluar_amod(tupla,oracion,oracionesTexto):
    pesoBase = 0
    if(tupla[2][0] in palabras_positivas):
        pesoBase = peso_palabras_positivas[palabras_positivas.index(tupla[2][0])]
        pesoBase += peso_adjetivo
        if oracion == oracionesTexto[0]:
            pesoBase = pesoBase + peso_oracion_inicial
        elif oracion == oracionesTexto[-1]:
            pesoBase = pesoBase + peso_oracion_final
        for tuplaNeg in oracion:
            if(tuplaNeg[1] == "neg"):
                pesoBase = 0 - pesoBase / 2
                break
        print("P")
        print(tupla, pesoBase)        

    elif(tupla[2][0] in palabras_negativas):
        pesoBase = peso_palabras_negativas[palabras_negativas.index(tupla[2][0])]
        pesoBase -= peso_adjetivo
        if oracion == oracionesTexto[0]:
            pesoBase = pesoBase - peso_oracion_inicial
        elif oracion == oracionesTexto[-1]:
            pesoBase = pesoBase - peso_oracion_final
        for tuplaNeg in oracion:
            if(tuplaNeg[1] == "neg"):
                pesoBase = 0 + pesoBase/4
                break
        print("N")
        print(tupla, pesoBase)
    return  pesoBase        
        
        
def evaluar_advmod(tupla,oracion, oracionesTexto):
    pesoBase = 0

    if(tupla[2][0] in palabras_positivas):
        pesoBase = peso_palabras_positivas[palabras_positivas.index(tupla[2][0])]                                 
        if oracion == oracionesTexto[0]:
            pesoBase = pesoBase + peso_oracion_inicial
        elif oracion == oracionesTexto[-1]:
            pesoBase = pesoBase + peso_oracion_final
        if tupla[0][0] in palabras_positivas and patternVerbo.match(tupla[0][1]):
            pesoBase = pesoBase + peso_verbos
        elif tupla[0][0] in palabras_negativas and patternVerbo.match(tupla[0][1]):
            pesoBase = pesoBase - peso_verbos
            
        for tuplaNeg in oracion:
            if(tuplaNeg[1] == "neg"):
                print(tuplaNeg)
                pesoBase = 0 - pesoBase/2
                break
        print("P")
        print(tupla, pesoBase)        
     
                                 
    elif(tupla[2][0] in palabras_negativas):
        pesoBase = peso_palabras_negativas[palabras_negativas.index(tupla[2][0])]
        if oracion == oracionesTexto[0]:
            pesoBase = pesoBase - peso_oracion_inicial
        elif oracion == oracionesTexto[-1]:
            pesoBase = pesoBase - peso_oracion_final
        if tupla[0][0] in palabras_positivas and patternVerbo.match(tupla[0][1]):
            pesoBase = pesoBase + peso_verbos
        elif tupla[0][0] in palabras_negativas and patternVerbo.match(tupla[0][1]):
            pesoBase = pesoBase - peso_verbos   
        
        for tuplaNeg in oracion:
            if(tuplaNeg[1] == "neg"):
                print(tuplaNeg)
                pesoBase = 0 + pesoBase/4
                break
        print("N")
        print(tupla, pesoBase)
    return pesoBase    

def evaluar_xcomp(tupla,oracion,oracionesTexto):
    pesoBase = 0
    if (tupla[2][0] in palabras_positivas):
        pesoBase = peso_palabras_positivas[palabras_positivas.index(tupla[2][0])]
        if oracion == oracionesTexto[0]:
            pesoBase = pesoBase + peso_oracion_inicial
        elif oracion == oracionesTexto[-1]:
            pesoBase = pesoBase + peso_oracion_final
        for tuplaNeg in oracion:
            if (tuplaNeg[1] == "neg"):
                pesoBase = 0 - pesoBase/2
                break
        print("P")
        print(tupla, pesoBase)


    elif(tupla[2][0] in palabras_negativas):
        pesoBase = peso_palabras_negativas[palabras_negativas.index(tupla[2][0])]
        if oracion == oracionesTexto[0]:
            pesoBase = pesoBase - peso_oracion_inicial
        elif oracion == oracionesTexto[-1]:
            pesoBase = pesoBase - peso_oracion_final
        for tuplaNeg in oracion:
            if (tuplaNeg[1] == "neg"):
                pesoBase = 0 + pesoBase / 4
                break
        print("N")
        print(tupla, pesoBase)
    return pesoBase

#Ejecucion desarrollo
    


#cargaDatos = pickle.load(open(pathDevPickle+corpus[0]+"/no9.txt",'rb'))

def cargarDatosDev():
    for c in corpus:
        pathDevCorpus = pathDevPickle + "/" +c
        for fichero in os.listdir(pathDevCorpus):   
            print(pathDevCorpus + "/"+fichero)
            oracionesTexto = pickle.load(open(pathDevCorpus + "/"+fichero,'rb'))
            pesoTotal = -50
            for oracion in oracionesTexto:
                for tupla in oracion:
                    if(tupla[1] == "amod"):
                        pesoTotal += evaluar_amod(tupla,oracion,oracionesTexto) #Evaluar adjetivos modificadores
        
                    if(tupla[1] == "advmod"): 
                        pesoTotal +=evaluar_advmod(tupla,oracion,oracionesTexto) #Tratamiento de adverbios 
                                 
                                 
                    if(tupla[1] == "xcomp"): 
                        pesoTotal +=evaluar_xcomp(tupla,oracion,oracionesTexto) #Tratamiento de xcomp
                        
                      
            if(pesoTotal > 0):
                resultado = fichero + "\t" +c+"\t"+"positive"
                listaResultados.append(resultado)
            else:
                resultado = fichero + "\t" +c+"\t"+"negative"
                listaResultados.append(resultado)  
                
#cargarDatosDev()

"""
for c in corpus2:
        pathDevCorpus = pathDev2 +"/"+c
        for fichero in os.listdir(pathDevCorpus):
            with open(pathDevCorpus +"/"+fichero,"r") as file:
                contents = Path(pathDevCorpus +"/"+fichero).read_text()
                remove_digits = str.maketrans('', '', digits)
                contents = contents.translate(remove_digits)
                print(pathDevCorpus +"/"+fichero)
            oracionesTexto = sentence_tokenizer.tokenize(contents)  #Obtenemos las diferentes oraciones que componen el texto
            pesoTotal = - 50
            
            for oracion in oracionesTexto:
                tokens_texto = whitespaceTokenizer.tokenize(contents)       #Tokenizar la frase
                [x.lower() for x in tokens_texto]                           #Transformar todas las palabras a minúscula para poder compararlas con el lexicón            
                tags = tagger.tag(tokens_texto)                             #Obtenemos las duplas palabra-etiqueta
                
                iterator = dependency_parser.raw_parse(oracion)
                for dep in iterator:
                    lista = list(dep.triples())
                    for tupla in lista:
                        if(tupla[1] == "amod"):
                            pesoTotal += evaluar_amod(tupla,oracion,lista) #Evaluar adjetivos modificadores
        
                        if(tupla[1] == "advmod"): 
                            pesoTotal +=evaluar_advmod(tupla,oracion,lista) #Tratamiento de adverbios 
                                 
                                 
                        if(tupla[1] == "xcomp"): 
                            pesoTotal +=evaluar_xcomp(tupla,oracion,lista) #Tratamiento de xcomp
                                                                                                 
                                  
                                 
            
            
            if(pesoTotal > 0):
                resultado = fichero + "\t" +c+"\t"+"positive"
                listaResultados.append(resultado)
            else:
                resultado = fichero + "\t" +c+"\t"+"negative"
                listaResultados.append(resultado)

"""
#print(listaResultados)
file = open('resultados.txt', 'w')
for fila in listaResultados:
    
    file.write(fila+"\n")

file.close()







