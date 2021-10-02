from flask import Flask, make_response, request, render_template, Response
import pandas as pd
#import csv 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#from matplotlib.figure import Figure
#import io
import seaborn as sns


app = Flask(__name__)

def load_data():
    df = pd.read_pickle('data/data.pkl')
    #data-preprocessing
    df.dropna(subset=['CNIM', 'CNAMA'], inplace=True)
    df.fillna(0, inplace=True)
    df.drop(columns=['STATUS', 'IPK','CSMTAWAL'], inplace=True)
    #add skewness
    ip = df[["IPS1","IPS2","IPS3","IPS4"]]
    ip_skew = ip.skew(axis=1)
    skew = pd.DataFrame({"SKEWNESS":ip_skew})
    df["SKEWNESS"] = ip_skew
    return df

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/transform', methods=["POST"])
def transform_view():
    if request.method == 'POST':
        f = request.files['data_file']
        df = pd.read_csv(f, delimiter=';', dtype={0:'string',1:'string',2:'string',3:'string',4:'string'})
        df.to_pickle("data/data.pkl")
        return render_template('simple.html',  tables=[df.head().to_html(classes='data')], titles=['sample data'])
        #return render_template('simple.html',  tables=df.head().to_html(header=False, classes='data'), titles=df.columns.values)
    return 'Oops, Try again something went wrong!'

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/dataprep', methods=['GET'])
def dataprep():
    #df = pd.read_pickle('data/data.pkl')

    #data-preprocessing
    #df.dropna(subset=['CNIM', 'CNAMA'], inplace=True)
    #df.fillna(0, inplace=True)
    #df.drop(columns=['STATUS', 'IPK','CSMTAWAL'], inplace=True)
    df = load_data()
    stats = df.describe().to_html(classes='table table-hover')

    #grafik rerata ips
    df_mean_ips = df[["IPS1","IPS2","IPS3","IPS4"]].mean()
    label = ["IPS1","IPS2","IPS3","IPS4"]
    plt.bar(label, list(df_mean_ips))
    plt.ylabel("Indeks Prestasi")
    plt.xlabel("Semester")
    plt.ylim([2, 4])
    plt.savefig('static/images/plot-rerata.png')
    plt.clf()

    #grafik per angkatan
    fig = plt.subplots(figsize=(12, 6))
    df_per_angkatan = df[["CTHAJARAWAL","CNIM"]].groupby(["CTHAJARAWAL"]).count().rename(columns={"CNIM":"JUMLAH"})
    plt.bar(df_per_angkatan.index.to_list(), df_per_angkatan["JUMLAH"])
    plt.ylabel("Jumlah Mahasiswa")
    plt.xlabel("Tahun Ajaran")
    plt.xticks(rotation=45)
    plt.savefig('static/images/plot-perangkatan.png')
    plt.clf()

    return render_template('dataprep.html', 
        statsdata=[ stats ],
    )

@app.route('/clustering', methods=['GET'])
def clustering():
    df = load_data()
    #clustering
    df_clust = df.drop(["CNAMA"], axis=1)
    X = df_clust.values[:,1:]
    from sklearn.preprocessing import StandardScaler
    clust_data = StandardScaler().fit_transform(X)

    from sklearn.cluster import KMeans 
    clusterNum = 3
    k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
    k_means.fit(X)
    labels = k_means.labels_
    df_labeled = df
    df_labeled["KLASTER"] = labels

    #rata-rata per klaster
    cluster_mean = df_labeled.groupby('KLASTER').mean()
    cluster_mean['JUMLAH'] = df_labeled['KLASTER'].value_counts()

    #grafik klaster
    klaster = df_labeled['KLASTER'].value_counts()
    plt.clf()
    plt.pie(
        list(klaster),
        labels=list(klaster.keys()),
        autopct='%1.1f%%',
    )
    plt.savefig('static/images/plot-klaster.png')

    return render_template('clustering.html', 
        clustermean = [ cluster_mean.to_html(classes='table table-hover')],
        klaster = list(klaster),
        klaster0 = [df_labeled[df_labeled['KLASTER'] == 0].head(10).to_html(classes='table table-hover')],
        klaster1 = [df_labeled[df_labeled['KLASTER'] == 1].head(10).to_html(classes='table table-hover')],
        klaster2 = [df_labeled[df_labeled['KLASTER'] == 2].head(10).to_html(classes='table table-hover')]
    )

#@app.route('/plot-rerata.png')
#def plot_rerata():
    #fig = create_figure()
    #fig = Figure()
    #df = pd.read_pickle('data/data.pkl')
    #rerata ips
    #df_mean_ips = df[["IPS1","IPS2","IPS3","IPS4"]].mean()
    #label = ["IPS1","IPS2","IPS3","IPS4"]
    #axis = fig.add_subplot(1, 1, 1)

    #axis.bar(label, df_mean_ips)
    #fig.ylabel("Indeks Prestasi")
    #fig.xlabel("Semester")
    #fig.ylim([2, 4])

    #output = io.BytesIO()
    #FigureCanvas(fig).print_png(output)
    #return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)