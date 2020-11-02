FROM python:3.7
EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
RUN python -m spacy download pt_core_news_sm && python -m spacy download en_core_web_sm
COPY . /app
CMD streamlit run app.py