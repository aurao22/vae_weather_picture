from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import re
import cv2
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import csv

def get_dir_files(dir_path, endwith=None, verbose=0):
    fichiers = None
    if endwith is not None:
        fichiers = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith(endwith)]
    else:
        fichiers = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    return fichiers

def get_regex_alphabetique_simple(verbose=0):
    pattern = re.compile(r'[^a-zA-Z]')
    return pattern

def get_numeric_columns_names(df, verbose=False):
    """Retourne les noms des colonnes numériques
    Args:
        df (DataFrame): Données
        verbose (bool, optional): Mode debug. Defaults to False.

    Returns:
        List(String): liste des noms de colonne
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    newdf = df.select_dtypes(include=numerics)
    return list(newdf.columns)

def get_outliers_datas(df, colname):
    """[summary]

    Args:
        df ([type]): [description]
        colname ([type]): [description]

    Returns:
        (float, float, float, float): q_low, q_hi,iqr, q_min, q_max
    """
    # .quantile(0.25) pour Q1
    q_low = df[colname].quantile(0.25)
    #  .quantité(0.75) pour Q3
    q_hi  = df[colname].quantile(0.75)
    # IQR = Q3 - Q1
    iqr = q_hi - q_low
    # Max = Q3 + (1.5 * IQR)
    q_max = q_hi + (1.5 * iqr)
    # Min = Q1 - (1.5 * IQR)
    q_min = q_low - (1.5 * iqr)
    return q_low, q_hi,iqr, q_min, q_max

def predict(n_clusters, kmeans, descriptor, model_MLPClassifier=None, verbose=0):
    nkp = len(descriptor)
    histo = np.zeros(n_clusters)

    for d in descriptor:
        idx = kmeans.predict([d])
        histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly
    
    probas = None
    if model_MLPClassifier is not None:
        probas = model_MLPClassifier.predict_proba([histo])

    return probas, histo


def pred_df(source_data_path, image_list, n_clusters, kmeans, mlp, species,img_descriptors_dic, label_encoder, file_name=None, verbose=0):
    df_test_dict =  defaultdict(list)
    if file_name is not None:
        result_file = open(source_data_path+ file_name, "w")
        result_file_obj = csv.writer(result_file)
        result_file_obj.writerow(np.append("id", species))

    histo_list = []
    pattern = get_regex_alphabetique_simple()

    for img_path in image_list:
        try:
            probas, histo = predict(n_clusters, kmeans, descriptor=img_descriptors_dic[img_path], model_MLPClassifier=mlp, verbose=verbose)
            proba = probas[0]
            histo_list.append(histo)

            if file_name is not None:
                row = []
                row.append(img_path)
                for e in probas[0]:
                    row.append(e)

                result_file_obj.writerow(row)

            name = img_path.lower().split(".")[0]
            name = re.sub(pattern, '', name).strip()
            
            df_test_dict["picture"].append(img_path)
            df_test_dict["target"].append(name)

            max_item = max(proba)
            pre = label_encoder.inverse_transform(np.where(proba == max_item)[0])[0]
            # pre = species[np.where(proba == max_item)][0]
            df_test_dict["predict"].append(pre)
            
            for i in range(0, len(proba)):
                label = label_encoder.inverse_transform([i])
                df_test_dict[label[0]+"_proba_"+str(i)].append(round(proba[i],2))  

        except Exception as error:
            print("ERROR : ", img_path)
            print(error)
    return df_test_dict, histo_list

def resize_picture(img, scale_percent=60, verbose=0):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized


def define_img_point(img_src_path, resize_scale_percent=None, display_cv2=False, display=True, nb_descriptors=300, verbose=0):
    #reading the image using imread() function from cv2 module and converting it into gray image
    grayimage = cv2.imread(img_src_path,0)
    if resize_scale_percent is not None:
        grayimage = resize_picture(grayimage, scale_percent=resize_scale_percent, verbose=verbose)
        
    #grayimage = cv2.cvtColor(readimage, cv2.COLOR_BGR2GRAY)
    #creating a sift object and using detectandcompute() function to detect the keypoints and descriptor from the image
    equ = cv2.equalizeHist(grayimage)

    siftobject = cv2.xfeatures2d.SIFT_create(nb_descriptors)
    keypoint, descriptor = siftobject.detectAndCompute(equ, None)
    #drawing the keypoints and orientation of the keypoints in the image and then displaying the image as the output on the screen
    keypointimage = cv2.drawKeypoints(equ, keypoint, grayimage, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if display:
        if display_cv2:
            cv2.imshow('SIFT', keypointimage)
            cv2.waitKey()
        else:
            plt.imshow(keypointimage)
            plt.title(f'img_kp size: {keypointimage.shape} - Gray: {grayimage.shape}') 
            plt.axis('off')
            plt.show()
    return grayimage, keypoint, descriptor


def transform_kp(file, scale_percent=None):
    gray= cv2.imread(file,0)
    if scale_percent is not None:
        gray = resize_picture(gray, scale_percent=scale_percent)
    equ = cv2.equalizeHist(gray)
    sift = cv2.SIFT_create(300)
    kp, des = sift.detectAndCompute(equ,None)
    img_kp =cv2.drawKeypoints(equ,kp,gray, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints.jpg',img_kp)
    print(f'Descripteurs :  {des.shape}')
    # print(des)

    # cv2.imshow('figure',img_kp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.imshow(img_kp)
    plt.title(f'img_kp size: {img_kp.shape} - Gray: {gray.shape}') 
    plt.axis('off')
    plt.show()
    return kp, des, img_kp


# ----------------------------------------------------------------------------------
#                        GRAPHIQUES
# ----------------------------------------------------------------------------------
PLOT_FIGURE_BAGROUNG_COLOR = 'white'
PLOT_BAGROUNG_COLOR = PLOT_FIGURE_BAGROUNG_COLOR

def color_graph_background(ligne=1, colonne=1):
    figure, axes = plt.subplots(ligne,colonne)
    figure.patch.set_facecolor(PLOT_FIGURE_BAGROUNG_COLOR)
    if isinstance(axes, np.ndarray):
        for axe in axes:
            # Traitement des figures avec plusieurs lignes
            if isinstance(axe, np.ndarray):
                for ae in axe:
                    ae.set_facecolor(PLOT_BAGROUNG_COLOR)
            else:
                axe.set_facecolor(PLOT_BAGROUNG_COLOR)
    else:
        axes.set_facecolor(PLOT_BAGROUNG_COLOR)
    return figure, axes


def graphe_outliers(df_out, column, q_min, q_max):
    """[summary]

    Args:
        df_out ([type]): [description]
        column ([type]): [description]
        q_min ([type]): [description]
        q_max ([type]): [description]
    """
    
    figure, axes = color_graph_background(1,2)
    # Avant traitement des outliers
    # Boite à moustaches
    #sns.boxplot(data=df_out[column],x=df_out[column], ax=axes[0])
    df_out.boxplot(column=[column], grid=True, ax=axes[0])
    # scatter
    df_only_ok = df_out[(df_out[column]>=q_min) & (df_out[column]<=q_max)]
    df_only_ouliers = df_out[(df_out[column]<q_min) | (df_out[column]>q_max)]
    plt.scatter(df_only_ok[column].index, df_only_ok[column].values, c='blue')
    plt.scatter(df_only_ouliers[column].index, df_only_ouliers[column].values, c='red')
    # Dimensionnement du graphe
    figure.set_size_inches(18, 7, forward=True)
    figure.set_dpi(100)
    figure.suptitle(column, fontsize=16)
    plt.show()




def show_hog(img_path, reduce_ratio=None, cmap="BrBG",orientations=9, pixels_per_cell=(8, 8)):
    img = imread(img_path)
     
    fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title(f'Original : {img.shape}')

    if reduce_ratio is not None:
        # resizing image
        resized_img = resize(img, (img.shape[0]/reduce_ratio, img.shape[1]/reduce_ratio))
    else:
        resized_img = img.copy()
    
    #creating hog features
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, channel_axis=-1)
    ax3.axis('off')
    ax3.imshow(hog_image, cmap=cmap)
    ax3.set_title(f'Hog Image : {hog_image.shape}')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=cmap)
    ax2.set_title(f'Hog Rescaled : {hog_image_rescaled.shape}')
    plt.show()
    # 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
    return img, resized_img, hog_image

def show_histogramme(img_param, equalize=False):
    dst = None
    img_l =img_param
    if isinstance(img_l, list) == False:
        img_l = [img_param]

    nb_graph = len(img_l)
    sub = int("2"+str(nb_graph)+"0")
    if equalize:
        sub = int(str(nb_graph)+"40")

    fig = plt.figure(figsize=(15, 5*nb_graph))

    for img in img_l:
        sub = _draw_hist_img(img=img, sub=sub, fig=fig)
        if equalize:
            dst = cv2.equalizeHist(img)
            sub = _draw_hist_img(img=dst, sub=sub, fig=fig)
    plt.show()
    return dst


def _draw_hist_img(img, sub, fig):
    sub += 1
    ax = fig.add_subplot(sub)
    ax.imshow(img)
    ax.axis('off')
    sub += 1

    ax = fig.add_subplot(sub)
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    ax.plot(cdf_normalized, color = 'b')
    ax.hist(img.flatten(),256,[0,256], color = 'r')
    ax.set_xlim([0,256])
    ax.legend(('cdf','histogram'), loc = 'upper left')
    return sub


def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.figure(figsize=(18,7), facecolor='white')
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=0.5, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            plt.figure(figsize=(10,10), facecolor="white")
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                valil = np.unique(illustrative_var)
                # On commence par traiter le NAN pour plus de lisibilité dans le graphe
                value = str(np.nan)
                if value in valil :
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, c=colors_dic(value, "blue"), s=100)
                    valil = valil[valil != value]
                for value in valil:
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, c=colors_dic.get(value, "blue"), s=100)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def _draw_hist_img_cv2(img, sub, fig):
    sub += 1
    cv2.imshow('figure',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sub += 1

    ax = fig.add_subplot(sub)
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    ax.plot(cdf_normalized, color = 'b')
    ax.hist(img.flatten(),256,[0,256], color = 'r')
    ax.set_xlim([0,256])
    ax.legend(('cdf','histogram'), loc = 'upper left')
    return sub

