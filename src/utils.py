
import sys, os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import numpy as np
import plotly.io as pio

import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
from matplotlib import cm as colours
from colorama import Fore, Back, Style
from scipy.stats import sem
import joblib
#import umap.umap_ as umap
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_samples
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from plotly.subplots import make_subplots
import pickle
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from sklearn.utils import shuffle

import librosa
import librosa.display


numtoPhon = {1:'a', 2:'ae', 3:'i', 4:'u', 5:'b', 6:'p', 7:'v', 8:'g', 9:'k'}
phoneme_group = {
        'a': 'low', 'ae': 'low', 'i': 'high', 'u': 'high',
        'b': 'labial', 'p': 'labial', 'v': 'labial',
        'g': 'dorsal', 'k': 'dorsal'
    }



# ================================================================
# GLOBAL PHONEME COLOR MAP (shared with boxplots)
# ================================================================

PHONEME_COLOR_MAP = {

    # HIGH — muted reds
    'i'  : 'rgba(135,38,38,1.0)',
    'u'  : 'rgba(172,55,55,1.0)',

    # LOW — muted oranges
    'a'  : 'rgba(197,123,34,1.0)',
    'ae' : 'rgba(226,161,71,1.0)',

    # DORSAL — blues
    'g'  : 'rgba(34,76,133,1.0)',
    'k'  : 'rgba(60,110,185,1.0)',

    # LABIAL — violets
    'b'  : 'rgba(93,60,136,1.0)',
    'p'  : 'rgba(124,87,173,1.0)',
    'v'  : 'rgba(156,120,197,1.0)'
}

def get_time(approx_time):
    # convert time seconds to mns+seconds
    val = approx_time/60
    rem = val - int(val)
    return f"{int(val)} mns and {rem * 60} seconds"
    # get_time(639)
    
def export_waveform_with_zoom(wav_path, t_start, t_end,
                              zoom_t_start, zoom_t_end,
                              outfile="waveform.svg"):
    y, sr = librosa.load(wav_path, sr=None)

    # main segment
    seg = y[int(t_start * sr): int(t_end * sr)]
    t_seg = np.linspace(t_start, t_end, len(seg))

    # zoom segment
    z = y[int(zoom_t_start * sr): int(zoom_t_end * sr)]
    t_z = np.linspace(zoom_t_start, zoom_t_end, len(z))

    fig = plt.figure(figsize=(12, 8), dpi=600)

    # main panel
    ax_main = fig.add_axes([0.1, 0.55, 0.8, 0.35])
    ax_main.plot(t_seg, seg, color="black", linewidth=0.5)
    ax_main.axis("off")

    # zoom panel
    ax_zoom = fig.add_axes([0.1, 0.1, 0.8, 0.35])
    ax_zoom.plot(t_z, z, color="black", linewidth=0.5)
    ax_zoom.axis("off")

    # response onset line 
    x = 0.5
    ax_main.plot([x, x], [0, 1], color="black", linewidth=0.5, transform=ax_main.transAxes)
    ax_zoom.plot([x, x], [1, 0], color="black", linewidth=0.5, transform=ax_zoom.transAxes)

    fig.savefig(outfile, format="svg", dpi=600, bbox_inches="tight", pad_inches=0)
    return fig

def export_spectrogram_with_zoom(wav_path, t_start, t_end,
                                 zoom_t_start, zoom_t_end,
                                 outfile="spectrogram.svg"):
    y, sr = librosa.load(wav_path, sr=None)

    # main segment spectrogram
    seg = y[int(t_start * sr): int(t_end * sr)]
    S = librosa.feature.melspectrogram(
        y=seg,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        power=2
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # zoom segment with finer temporal resolution
    z = y[int(zoom_t_start * sr): int(zoom_t_end * sr)]
    S_z = librosa.feature.melspectrogram(
        y=z,
        sr=sr,
        n_fft=512,
        hop_length=32,
        n_mels=128,
        power=2
    )
    S_z_db = librosa.power_to_db(S_z, ref=np.max)

    fig = plt.figure(figsize=(12, 8), dpi=600)

    ax_main = fig.add_axes([0.1, 0.55, 0.8, 0.35])
    librosa.display.specshow(S_db, sr=sr, hop_length=256, cmap="magma", ax=ax_main)
    ax_main.axis("off")

    ax_zoom = fig.add_axes([0.1, 0.1, 0.8, 0.35])
    librosa.display.specshow(S_z_db, sr=sr, hop_length=32, cmap="magma", ax=ax_zoom)
    ax_zoom.axis("off")

    # simple connector line (optional, customize later)
    x = 0.5
    ax_main.plot([x, x], [0, 1], color="black", linewidth=0.5, transform=ax_main.transAxes)
    ax_zoom.plot([x, x], [1, 0], color="black", linewidth=0.5, transform=ax_zoom.transAxes)

    fig.savefig(outfile, format="svg", dpi=600, bbox_inches="tight", pad_inches=0)
    return fig

def export_spectrogram(wav_path, t_start, t_end, outfile="spectrogram.svg"):
    y, sr = librosa.load(wav_path, sr=None)
    segment = y[int(t_start * sr): int(t_end * sr)]

    S = librosa.feature.melspectrogram(
        y=segment, sr=sr,
        n_fft=1024, hop_length=256,
        n_mels=128, power=2
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(12, 5), dpi=600)
    librosa.display.specshow(S_db, sr=sr, hop_length=256, cmap="magma")
    plt.axis("off")

    plt.savefig(outfile, format="svg", dpi=600, bbox_inches="tight", pad_inches=0)
    plt.show()
    plt.close()
    return 





################################################################
##################### Utils for figure 2   #####################
################################################################

# ================================================================
# PHONEME METADATA
# ================================================================

PHONEMES = {
    'a':  {'vc': 'vowel',     'artic': 'low'},
    'ae': {'vc': 'vowel',     'artic': 'low'},
    'i':  {'vc': 'vowel',     'artic': 'high'},
    'u':  {'vc': 'vowel',     'artic': 'high'},
    'b':  {'vc': 'consonant', 'artic': 'labial'},
    'p':  {'vc': 'consonant', 'artic': 'labial'},
    'v':  {'vc': 'consonant', 'artic': 'labial'},
    'g':  {'vc': 'consonant', 'artic': 'dorsal'},
    'k':  {'vc': 'consonant', 'artic': 'dorsal'}
}

ARTIC_ORDER = ['high','low','dorsal','labial']


# ================================================================
# PHONEME COLORS (muted categorical)
# ================================================================

PHONEME_COLOR_MAP = {

    # HIGH — muted reds
    'i'  : 'rgba(135,38,38,1.0)',
    'u'  : 'rgba(172,55,55,1.0)',

    # LOW — muted oranges
    'a'  : 'rgba(197,123,34,1.0)',
    'ae' : 'rgba(226,161,71,1.0)',

    # DORSAL — blues
    'g'  : 'rgba(34,76,133,1.0)',
    'k'  : 'rgba(60,110,185,1.0)',

    # LABIAL — violets
    'b'  : 'rgba(93,60,136,1.0)',
    'p'  : 'rgba(124,87,173,1.0)',
    'v'  : 'rgba(156,120,197,1.0)'
}

orderforplot = {
    "Difference_Variable": ['Onset', 'Offset', 'Duration'], 
    "Phoneme": ['a', 'ae', 'b', 'g', 'i', 'k', 'p', 'u', 'v'], 
    "Patient": ['S1', 'S2', 'S3', 'S4']    
}


def make_phoneme_colormap(_):
    return PHONEME_COLOR_MAP




def plotly_general(df, y, width, height, name, colorer=None, x=None, facet_col=None, jitter=0.4, boxgroupgap=0.8, boxgap=0.8, save=True):

    if x is not None:
        boxgroupgap=0.6
    fig = px.box(df, y=y, x=x, facet_col=facet_col, color=colorer, points="all",
                 boxmode="group",
                 hover_name='Trial',
                 hover_data=['ApproxTime','Time', 'Patient', 'Phoneme', 'Position', 'Vowel/Consonant','Articulatory Group'],

                 color_discrete_sequence=px.colors.qualitative.Plotly,
                 category_orders=orderforplot)

    fig.update_traces(marker=dict(size=1.6, opacity=0.4, line=dict(width=0.2, color="#d1cfcf")),
                      jitter=jitter, boxmean=True, opacity=1.0)

    fig.update_layout(title="Phoneme Time Differences: Manual vs. Automated",
                      title_font=dict(size=7, family="Arial", color="black"),
                      font=dict(family="Arial", size=6, color="black"),
                      yaxis_title="Time (s)", plot_bgcolor="white", paper_bgcolor="white",
                      boxgroupgap=boxgroupgap, boxgap=boxgap, margin=dict(t=40, b=40, l=50, r=10),
                      legend=dict(title_text=None, font=dict(size=6)))
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", xaxis_showgrid=False, yaxis_showgrid=False)

    fig.update_xaxes(showticklabels=True, showline=True, linecolor="black", mirror=False,
                     ticks="outside", tickfont=dict(size=5, family="Arial"),
                     title_font=dict(size=6, family="Arial"))
    fig.update_yaxes(showticklabels=True, showline=True, linecolor="black", mirror=False,
                     ticks="outside", tickfont=dict(size=5, family="Arial"),
                     title_font=dict(size=6, family="Arial"))

    if save:
        fig.write_image(f"../figures/figure2/fig1_{name}.svg", 
                    format="svg", scale=3)
    fig.show()
    return None


# ================================================================
# PLOTTING FUNCTION
# ================================================================

def plotly_phonemes(df, save=True):

    linewidth = 0.67              # ~= 0.5 pt for Illustrator
    jitter = 1.0                 # max allowed
    pointpos = -1.30             # visual jitter extension
    boxgap = 0.25                # thicker boxes
    boxgroupgap = 0.40

    phoneme_order = []
    for a in ARTIC_ORDER:
        phoneme_order += [p for p,d in PHONEMES.items() if d['artic']==a]

    fig = px.box(
        df,
        y="Time",
        color="Phoneme",
        points="all",
        boxmode="group",
        hover_name='Trial',
        hover_data=[
            'ApproxTime','Time','Patient','Phoneme','Position',
            'Vowel/Consonant','Articulatory Group'
        ],
        color_discrete_map=PHONEME_COLOR_MAP,
        category_orders={"Phoneme": phoneme_order}
    )

    # ------------------------
    # TRACES
    # ------------------------
    fig.update_traces(
        marker=dict(size=1.6, opacity=0.45,
                    line=dict(width=0.15, color="silver")),
        jitter=jitter,
        pointpos=pointpos,
        boxmean=False,          
        opacity=1.0
    )

    # ------------------------
    # AXES
    # ------------------------
    fig.update_xaxes(
        showticklabels=False,
        showline=False,
        ticks=""
    )

    fig.update_yaxes(
        showticklabels=True,
        showline=True,
        linecolor="black",
        linewidth=linewidth,
        ticks="outside",
        tickfont=dict(size=5, family="Arial"),
        title_font=dict(size=6, family="Arial")
    )

    # ------------------------
    # LAYOUT
    # ------------------------
    fig.update_layout(
        title="Phoneme Time Differences: Manual vs. Automated",
        title_font=dict(size=7, family="Arial", color="black"),
        font=dict(family="Arial", size=6, color="black"),
        yaxis_title="Time (s)",

        plot_bgcolor="white",
        paper_bgcolor="white",

        boxgap=boxgap,
        boxgroupgap=boxgroupgap,

        margin=dict(t=40, b=45, l=50, r=10),

        legend=dict(
            orientation="h",
            y=-0.20,
            x=0.5,
            xanchor="center",
            font=dict(size=6)
        )
    )

    if save:
        fig.write_image(
            "../figures/figure2/fig1_by_phoneme.svg",
            format="svg"
        )

    fig.show()




################################################################
##################### Utils for decoding   #####################
################################################################

class PCA_noCenter(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = self._get_components(X, S)
        self.components_ = Vt[:k].T
        self.explained_variance_ = S**2
        return self

    def transform(self, X):
        return X @ self.components_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def _get_components(self, X, S):
        if self.n_components is None or self.n_components >= min(X.shape):
            print("n_components is None or greater than the number of features/samples. Using n_components = min(X.shape)")
            return min(X.shape)
        elif self.n_components < 1:
            cum_var = np.cumsum(S**2) / np.sum(S**2)
            return np.argmax(cum_var >= self.n_components) + 1
        else:
            return int(self.n_components)
        
        
        
        
        




################################################################
##################### Utils for figure 3   #####################
################################################################


def phoneme_type(phoneme):

    consonants = ['b', 'p', 'v', 'g', 'k']
    vowels = ['a', 'ae', 'i', 'u']

    if phoneme.lower() in consonants:
        return "Consonant"
    elif phoneme.lower() in vowels:
        return "Vowel"
    else:
        raise ValueError("Phoneme not in categories")

def get_position_index(position: str) -> int:
    position_map = {'p1': 0, 'p2': 1, 'p3': 2}
    if position not in position_map:
        raise ValueError(f"Invalid position: {position}")
    return position_map[position]

def getCoolData(patient_name, position, method, pkl_path, intraop_path):
    
    patient_data = fetch_patient_data(pkl_path=pkl_path, intra_op_data_path=intraop_path)
    timevec = np.arange(-0.5, 0.5, 1/200)

    if position == 'all':
        X_train = np.concatenate(
            (
            patient_data[patient_name]['p1'][method]['hg_trace'], 
            patient_data[patient_name]['p2'][method]['hg_trace'], 
            patient_data[patient_name]['p3'][method]['hg_trace']
            ), 
            axis=0
        )
        y_train = np.concatenate(
            (
            patient_data[patient_name]['p1'][method]['phon_seq'][:, 0],
            patient_data[patient_name]['p2'][method]['phon_seq'][:, 1],
            patient_data[patient_name]['p3'][method]['phon_seq'][:, 2]
            ), 
            axis=0
        )
        tr, tim, chan = X_train.shape
        X_train = X_train.transpose(0, 2, 1) 
        
        y_train_positions = np.concatenate(
            [
            ['p1']*len(patient_data[patient_name]['p1'][method]['phon_seq'][:, 0]),
            ['p2']*len(patient_data[patient_name]['p1'][method]['phon_seq'][:, 1]),
            ['p3']*len(patient_data[patient_name]['p1'][method]['phon_seq'][:, 2])
            ],
            axis=0)
    else:
        X_train = patient_data[patient_name][position][method]['hg_trace']
        tr, tim, chan = X_train.shape
        X_train = X_train.transpose(0, 2, 1) 
        y_train = patient_data[patient_name][position][method]['phon_seq'][:, get_position_index(position)]
        y_train_positions = np.array([position]*len(y_train))
    
    y_train_phon = np.array([numtoPhon[i] for i in y_train])
    y_ph_group = np.array([phoneme_group[i] for i in y_train_phon])
    y_ph_type = np.array([phoneme_type(i) for i in y_train_phon])
    
    return timevec, X_train, y_train, y_train_phon, y_ph_type, y_ph_group, y_train_positions

def fetch_patient_data(pkl_path, intra_op_data_path):
    
    """provide raw data if pkl not available"""
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            patient_data = pickle.load(f)
        print("Loaded patient_data from pickle.")
    else:
        # NOTE: Raw .mat files are stored locally and not tracked in this repo.
        # If patient_data.pkl does not exist, update this path to the location
        # of your raw data directory containing per-patient .mat files.
        patients = ['S14', 'S22', 'S23', 'S26', 'S33']
        positions = ['p1', 'p2', 'p3']

        patient_data = {}

        for patient in patients:
            patient_data[patient] = {}
            for position in positions:
                patient_data[patient][position] = {'Kumar': {}, 'MFA': {}}

                data_kumar = sp.io.loadmat(f"{intra_op_data_path}\\{patient}\\{patient}_HG_{position}_sigChannel_goodTrials.mat")
                data_mfa = sp.io.loadmat(f"{intra_op_data_path}\\{patient}\\{patient}_HG_{position}_sigChannel_goodTrials_MFA.mat")

                patient_data[patient][position]['Kumar'] = {
                    'hg_trace': data_kumar['hgTrace'],
                    'hg_map': data_kumar['hgMap'],
                    'phon_seq': data_kumar['phonSeqLabels']
                }

                patient_data[patient][position]['MFA'] = {
                    'hg_trace': data_mfa['hgTrace'],
                    'hg_map': data_mfa['hgMap'],
                    'phon_seq': data_mfa['phonSeqLabels']
                }

        # Save to pickle
        with open(pkl_path, "wb") as f:
            pickle.dump(patient_data, f)

        print("Processed and saved patient_data to pickle.")
    return patient_data



def get_training_data(patient_data, patient_name, position, method):
    if position == 'all':
        X_train = np.concatenate(
            (
            patient_data[patient_name]['p1'][method]['hg_trace'],
            patient_data[patient_name]['p2'][method]['hg_trace'],
            patient_data[patient_name]['p3'][method]['hg_trace']
            ),
            axis=0
        )
        y_train = np.concatenate(
            (
            patient_data[patient_name]['p1'][method]['phon_seq'][:, 0],
            patient_data[patient_name]['p2'][method]['phon_seq'][:, 1],
            patient_data[patient_name]['p3'][method]['phon_seq'][:, 2]
            ),
            axis=0
        )

        X_train = X_train.reshape(X_train.shape[0], -1)
        y_train_positions = np.concatenate(
            [
            ['p1']*len(patient_data[patient_name]['p1'][method]['phon_seq'][:, 0]),
            ['p2']*len(patient_data[patient_name]['p1'][method]['phon_seq'][:, 1]),
            ['p3']*len(patient_data[patient_name]['p1'][method]['phon_seq'][:, 2])
            ],
            axis=0)
    else:
        X_train = patient_data[patient_name][position][method]['hg_trace']
        X_train = X_train.reshape(X_train.shape[0], -1)
        y_train = patient_data[patient_name][position][method]['phon_seq'][:, get_position_index(position)]
        y_train_positions = np.array([position]*len(y_train))

    y_train_phon = np.array([numtoPhon[i] for i in y_train])
    y_ph_group = np.array([phoneme_group[i] for i in y_train_phon])
    y_ph_type = np.array([phoneme_type(i) for i in y_train_phon])
    Xys_df = pd.DataFrame({
        'X_train': list(X_train),  # each row is a feature vector
        'y_train_num': y_train,
        'y_train_phon': y_train_phon,
        'y_train_phontype': y_ph_type,
        'y_train_phongrp': y_ph_group,
        'y_train_pos': y_train_positions,
        'Method': [method] * len(y_train),
    })
    Xys_df['Patient'] = patient_name
    return Xys_df


def run_tsne(Xys_df, perp, pcacomp, numrun, pca_ref=None):

    tsne_model = TSNE(n_components=2, init='random', perplexity=perp, n_jobs=-1, random_state=numrun)

    X_train = np.stack(Xys_df['X_train'].values)
    y_train = Xys_df['y_train_num'].to_numpy()
    y_train_pos = Xys_df['y_train_pos'].to_numpy()
    y_train_type = Xys_df['y_train_phontype'].to_numpy()
    y_train_phgrp = Xys_df['y_train_phongrp'].to_numpy()

    if pca_ref is None:
        pca_model = PCA_noCenter(n_components=pcacomp)
        X_pca = pca_model.fit_transform(X_train)
        pca_ref = pca_model
    else:
        X_pca = pca_ref.transform(X_train)

    X_tsne = tsne_model.fit_transform(X_pca)

    tsne_df = pd.DataFrame(X_tsne, columns=['tsne-1', 'tsne-2'])
    tsne_df['Phoneme'] = Xys_df['y_train_phon']
    tsne_df['Phoneme_Position'] = Xys_df['y_train_pos']
    tsne_df['Phoneme_Type'] = Xys_df['y_train_phontype']
    tsne_df['Phoneme_Group'] = Xys_df['y_train_phongrp']
    tsne_df['Perplexity'] = perp
    tsne_df['KL_Divergence'] = tsne_model.kl_divergence_
    tsne_df['Silhoutte_Score_Phon'] = np.mean(np.array(silhouette_samples(X_tsne, y_train))[(np.array(silhouette_samples(X_tsne, y_train)) > 0)])
    if tsne_df['Phoneme_Position'].nunique() > 2:
        tsne_df['Silhoutte_Score_PhonPos'] = np.mean(np.array(silhouette_samples(X_tsne, y_train_pos))[(np.array(silhouette_samples(X_tsne, y_train_pos)) > 0)])
    tsne_df['Silhoutte_Score_PhonType'] = np.mean(np.array(silhouette_samples(X_tsne, y_train_type))[(np.array(silhouette_samples(X_tsne, y_train_type)) > 0)])
    tsne_df['Silhoutte_Score_PhonGrp'] = np.mean(np.array(silhouette_samples(X_tsne, y_train_phgrp))[(np.array(silhouette_samples(X_tsne, y_train_phgrp)) > 0)])
    tsne_df['Patient'] = Xys_df['Patient'].iloc[0]
    tsne_df['Method'] = Xys_df['Method'].iloc[0]
    tsne_df['#Run'] = numrun
    if tsne_df['Phoneme_Position'].nunique() > 2:
        tsne_df = tsne_df[['#Run', 'tsne-1', 'tsne-2', 'Phoneme', 'Phoneme_Position', 'Phoneme_Type',
                           'Phoneme_Group', 'Perplexity', 'KL_Divergence','Silhoutte_Score_Phon',
                           'Silhoutte_Score_PhonPos', 'Silhoutte_Score_PhonType',
                           'Silhoutte_Score_PhonGrp', 'Patient', 'Method']]
    else:
        tsne_df = tsne_df[['#Run', 'tsne-1', 'tsne-2', 'Phoneme', 'Phoneme_Position', 'Phoneme_Type',
                           'Phoneme_Group', 'Perplexity', 'KL_Divergence','Silhoutte_Score_Phon',
                           'Silhoutte_Score_PhonType', 'Silhoutte_Score_PhonGrp','Patient', 'Method']]
    return tsne_df, pca_ref


def run_tsneShuffledY(Xys_df, perp, pcacomp, numrun):
    tsne_model = TSNE(n_components=2, init='random', perplexity=perp, n_jobs=-1, random_state=numrun)
    pca_model = PCA_noCenter(n_components=pcacomp)

    X_train = np.stack(Xys_df['X_train'].values)
    y_train = Xys_df['y_train_num'].to_numpy()
    # to create a chance silhouette distribution, we will shuffle the labels
    y_trainChance = shuffle(y_train, random_state=numrun)
    y_train_pos = Xys_df['y_train_pos'].to_numpy()
    y_train_type = Xys_df['y_train_phontype'].to_numpy()
    y_train_phgrp = Xys_df['y_train_phongrp'].to_numpy()

    X_pca = pca_model.fit_transform(X_train)
    X_tsne = tsne_model.fit_transform(X_pca)

    tsne_df = pd.DataFrame(X_tsne, columns=['tsne-1', 'tsne-2'])
    tsne_df['Phoneme'] = Xys_df['y_train_phon']
    tsne_df['Phoneme_Position'] = Xys_df['y_train_pos']
    tsne_df['Phoneme_Type'] = Xys_df['y_train_phontype']
    tsne_df['Phoneme_Group'] = Xys_df['y_train_phongrp']
    tsne_df['Perplexity'] = perp
    tsne_df['KL_Divergence'] = tsne_model.kl_divergence_
    tsne_df['Silhoutte_Score_Phon'] = np.mean(np.array(silhouette_samples(X_tsne, y_train))[(np.array(silhouette_samples(X_tsne, y_train)) > 0)])
    tsne_df['Silhouette_Score_PhonChance'] = np.mean(np.array(silhouette_samples(X_tsne, y_trainChance))[(np.array(silhouette_samples(X_tsne, y_trainChance)) > 0)])
    if tsne_df['Phoneme_Position'].nunique() > 2:
        tsne_df['Silhoutte_Score_PhonPos'] = np.mean(np.array(silhouette_samples(X_tsne, y_train_pos))[(np.array(silhouette_samples(X_tsne, y_train_pos)) > 0)])
    tsne_df['Silhoutte_Score_PhonType'] = np.mean(np.array(silhouette_samples(X_tsne, y_train_type))[(np.array(silhouette_samples(X_tsne, y_train_type)) > 0)])
    tsne_df['Silhoutte_Score_PhonGrp'] = np.mean(np.array(silhouette_samples(X_tsne, y_train_phgrp))[(np.array(silhouette_samples(X_tsne, y_train_phgrp)) > 0)])
    tsne_df['Patient'] = Xys_df['Patient'].iloc[0]
    tsne_df['Method'] = Xys_df['Method'].iloc[0]
    tsne_df['#Run'] = numrun
    if tsne_df['Phoneme_Position'].nunique() > 2:
        tsne_df = tsne_df[['#Run', 'tsne-1', 'tsne-2', 'Phoneme', 'Phoneme_Position', 'Phoneme_Type',
                           'Phoneme_Group', 'Perplexity', 'KL_Divergence',
                           'Silhoutte_Score_Phon', 'Silhouette_Score_PhonChance',
                           'Silhoutte_Score_PhonPos', 'Silhoutte_Score_PhonType',
                           'Silhoutte_Score_PhonGrp', 'Patient', 'Method']]
    else:
        tsne_df = tsne_df[['#Run', 'tsne-1', 'tsne-2', 'Phoneme', 'Phoneme_Position', 'Phoneme_Type',
                           'Phoneme_Group', 'Perplexity', 'KL_Divergence',
                           'Silhoutte_Score_Phon', 'Silhouette_Score_PhonChance',
                           'Silhoutte_Score_PhonType', 'Silhoutte_Score_PhonGrp','Patient', 'Method']]
    return tsne_df


def run_plottly(tsne_df, huere):

    pio.templates.default = "none"

    catorder = None
    color_map = None

    if huere == 'Phoneme':

        silscore = tsne_df['Silhoutte_Score_Phon'].iloc[0]

        # LOCKED color mapping — no randomness
        color_map = PHONEME_COLOR_MAP

        # Stable phoneme display order
        catorder = {
            "Phoneme": ['i','u','a','ae','g','k','b','p','v']
        }

    elif huere == 'Phoneme_Position':

        silscore = tsne_df['Silhoutte_Score_PhonPos'].iloc[0]
        # Default palette is fine for this diagnostic plot
        color_map = None

    elif huere == 'Phoneme_Type':

        silscore = tsne_df['Silhoutte_Score_PhonType'].iloc[0]
        color_map = None

    elif huere == 'Phoneme_Group':

        silscore = tsne_df['Silhoutte_Score_PhonGrp'].iloc[0]
        color_map = None

    else:
        raise ValueError(
            "Invalid hue parameter. Choose from 'Phoneme', "
            "'Phoneme_Position', 'Phoneme_Type', or 'Phoneme_Group'."
        )

    numrun = tsne_df['#Run'].iloc[0]
    method = tsne_df['Method'].iloc[0]
    patient = tsne_df['Patient'].iloc[0]

    # ------------------------------------------------------------
    # PLOT
    # ------------------------------------------------------------

    fig = px.scatter(
        tsne_df,
        x='tsne-1',
        y=-tsne_df['tsne-2'],
        color=huere,
        opacity=0.7,

        # << FIX: consistent phoneme colors >>
        color_discrete_map=color_map,
        category_orders=catorder
    )

    fig.update_layout(
        title=(
            f"TSNE Colored on: {huere} "
            f"- Silhouette Score: {silscore:.3f} "
            f"- Patient: {patient} "
            f"- Method: {method} "
            f"#Run: {numrun}"
        ),
        legend_title_text=huere
    )

    fig.update_xaxes(showticklabels=False, showline=True, linecolor='black')
    fig.update_yaxes(showticklabels=False, showline=True, linecolor='black')

    if silscore > 0.25:
        out_dir = os.path.join("..", "figures", "figure3", "TSNE_Clusters")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"TSNE on {huere} Silscore {silscore:.2f} {patient} {method} {numrun}.svg")
        fig.write_image(out_path, format="svg", scale=3)


    return fig



def fetch_silscore_data(patient_data,
                        pkl_path=r"../data/processed/silscores_allPatients.pkl",
                        patients=None,
                        methods=None,
                        position="p1",
                        perp_default=50,
                        new_run= True, 
                        num_iters=50):

    if patients is None:
        patients = ['S26', 'S22', 'S23', 'S33']

    if methods is None:
        methods = ['Kumar', 'MFA']

    if os.path.exists(pkl_path) and not new_run:
        with open(pkl_path, "rb") as f:
            silscores = pickle.load(f)
        return silscores

    silscores = {}

    for patient in patients:
        silscores[patient] = {}

        for method in methods:
            silscores[patient][method] = {
                'Silhoutte_Score_PhonType': [],
                'Silhoutte_Score_PhonGrp': [],
                'Silhoutte_Score_Phon': [],
                'Silhouette_Score_PhonChance': []
            }

            Xys_df = get_training_data(patient_data, patient, position, method)

            perp = 30 if patient == 'S33' else perp_default

            for i in range(1, num_iters + 1):
                tsne_df = run_tsneShuffledY(Xys_df, perp, 0.8, i)

                silscores[patient][method]['Silhoutte_Score_PhonType'].append(
                    tsne_df['Silhoutte_Score_PhonType'].iloc[0]
                )
                silscores[patient][method]['Silhoutte_Score_PhonGrp'].append(
                    tsne_df['Silhoutte_Score_PhonGrp'].iloc[0]
                )
                silscores[patient][method]['Silhoutte_Score_Phon'].append(
                    tsne_df['Silhoutte_Score_Phon'].iloc[0]
                )
                silscores[patient][method]['Silhouette_Score_PhonChance'].append(
                    tsne_df['Silhouette_Score_PhonChance'].iloc[0]
                )

    with open(pkl_path, "wb") as f:
        pickle.dump(silscores, f)

    print("Saved silscores to pickle.")
    return silscores


def mask_phoneme_channel(X, y, phoneme, channel):
    # keep the indexing of the trials, we need that 
    mask = (y == phoneme)
    # get the indices of the trials where phoneme matches
    indices = np.where(mask)[0]
    if channel == "all":
        X = X[mask, :, :]
    else:
        X = X[mask, channel, :]
    y = y[mask]
    return X, y, indices


def rgb_string_to_rgba(rgb_string, alpha):
    nums = rgb_string.strip('rgb()').split(',')
    r, g, b = [int(n) for n in nums]
    return f'rgba({r}, {g}, {b}, {alpha})'


def silscores_to_df(silscores):
    rows = []
    for patient, methods in silscores.items():
        for method, scores in methods.items():
            for score_type, values in scores.items():
                for iteration, value in enumerate(values, start=1):
                    rows.append({
                        'Patient':    patient,
                        'Method':     method,
                        'Score_Type': score_type,
                        'Iteration':  iteration,
                        'Value':      value,
                    })
    silscores_df = pd.DataFrame(rows)
    silscores_df = silscores_df[["Patient", "Method", "Score_Type", "Iteration", "Value"]]
    silscores_df.sort_values(by=["Patient", "Method", "Score_Type", "Iteration"], inplace=True)
    return silscores_df






