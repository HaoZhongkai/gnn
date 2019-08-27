from util import *
import pickle
import os

workdir = os.path.dirname(__file__)

dataset_dict = {
    'COLLAB':'COLLAB',
    'IMDB-B':'IMDBBINARY',
    'IMDB-M':'IMDBMULTI',
    'MUTAG':'MUTAG',
    'NCl1':'NCl1',
    'PROTEINS':'PROTEINS',
    'PTC':'PTC',
    'RDT-B':'REDDITBINARY',
    'RDT-M5K':'REDDITMULTI5K'
}
dataset_name = ['COLLAB','IMDBBINARY','IMDBMULTI','MUTAG','NCl1','PROTEINS',
                'PTC','REDDITBINARY','REDDITMULTI5K']

def preprocess(dataset, degree_as_tag=True, embedded_dim=4):
    g_list, label_num = load_data(dataset, degree_as_tag,embedded_dim)
    pickle.dump((g_list,label_num),open(workdir+'/dataset/{}_{}.pkl'.format(dataset,embedded_dim),'wb'))


preprocess(dataset_dict['RDT-B'],True,6)