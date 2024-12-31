# movie_searcher_recommender
Provides a synopsis about movie requested and recommends a playlist of other movies other users have viewed

Implements:
- ALS trained model in Spark for recommendations
- LangChain to prompt LLM via Groq https://python.langchain.com/docs/integrations/chat/groq/
- Flask in Python to user interface https://flask.palletsprojects.com/en/stable/

Steps:

1) run a docker image
   $ docker container run -d -p 5000:5000 -p 4040:4040 python:3.10 sleep infinity
   
2) pip install -r requirements.txt

3) setup java install

   git clone https://github.com/mkenjis/apache_binaries
   
   tar zxvf jre-8u181-linux-x64.tar.gz
   
   cat >>.bashrc
   export JAVA_HOME=/usr/local/jre1.8.0_181
   export CLASSPATH=$JAVA_HOME/lib
   export PATH=$PATH:.:$JAVA_HOME/bin
   <Ctrl-D>

   . .bashrc
   
   java -version
   java version "1.8.0_181"
   Java(TM) SE Runtime Environment (build 1.8.0_181-b13)
   Java HotSpot(TM) 64-Bit Server VM (build 25.181-b13, mixed mode)

3) python init_db.py

4) python recomm_system.py


NOTICE: ALS trained model is created following the steps_to_test_als_model.txt script file in pyspark.