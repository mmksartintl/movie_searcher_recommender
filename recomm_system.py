
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").getOrCreate()

from pyspark.ml.recommendation import ALSModel

alsmodel = ALSModel.load("./recomm_system")

from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np

def cos_sim(a,b):
    return float(np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)))

cossim = udf(cos_sim, DoubleType())

item = spark.read.option("inferSchema","true").option("sep","|").csv("./movielens/u.item").toDF("id","title","rel_date","dum1","link","cat1","cat2","cat3","cat4","cat5","cat6","cat7","cat8","cat9","cat10","cat11","cat12","cat13","cat14","cat15","cat16","cat17","cat18","cat19")

def get_similar_movies(movie_id):
    moviesel_df = alsmodel.itemFactors.where(col("id") == movie_id).select(col("features").alias("features_sel"))

    moviecross_df = moviesel_df.crossJoin(alsmodel.itemFactors)

    moviecs_df = moviecross_df.withColumn('cs',cossim("features_sel","features"))

    moviesim_df = moviecs_df.join(item, "id").select("cs","title").orderBy(desc("cs")).limit(10).collect()

    return moviesim_df


from langchain_groq import ChatGroq

llm = ChatGroq(temperature=0,groq_api_key='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',
               model_name="llama3-8b-8192")

#page_data = moviecs_df.join(item, "id").select("title").orderBy(desc("cs")).limit(1).collect()

from langchain_core.prompts import PromptTemplate

prompt_extract = PromptTemplate.from_template(
    """
    ### SCRAPED TEXT FROM WEBSITE:
    {page_data}
    ### INSTRUCTION:
    1) Bring me a sinopse about movie.
    2) List the director.
    3) List the actors / actress.
    4) List the movie type.
    5) List related movies.
    ### NO PREAMBLE
    """
)

chain_extract = prompt_extract | llm

#res = chain_extract.invoke(input={'page_data': page_data})
#print(res.content)

import sqlite3
from flask import Flask, render_template, request, url_for, flash, redirect

app = Flask(__name__)


def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/')
def index():
    conn = get_db_connection()
    posts = conn.execute('SELECT * FROM posts').fetchall()
    post1 = conn.execute('SELECT * FROM posts').fetchone()
    conn.close()
    if posts:
       movie_title = post1['title']
       combined_string = f"%{movie_title}%"
       moviesel_df = item.where(col("title").like(combined_string)).select("id","title").first()
       if moviesel_df:
          movies = get_similar_movies(moviesel_df.id)
       else:
          movies = get_similar_movies(0)
    else:
       movies = get_similar_movies(0)
    return render_template('index.html',posts=posts,movies=movies)


@app.route('/create', methods=('GET', 'POST'))
def create():
    # if the user clicked on Submit, it sends post request
    if request.method == 'POST':
        # Get the title and save it in a variable
        title = request.form['title']
        # Get the content the user wrote and save it in a variable
        #content = request.form['content']
        if not title:
            flash('Title is required!')
        else:
            res = chain_extract.invoke(input={'page_data': title})
            content = res.content
            # Open a connection to databse
            conn = get_db_connection()
            # Insert the new values in the db
            conn.execute('DELETE FROM posts')
            conn.execute('INSERT INTO posts (title, content) VALUES (?, ?)',(title, content))
            conn.commit()
            conn.close()
            # Redirect the user to index page
            return redirect(url_for('index'))

    return render_template('create.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    spark.stop()
