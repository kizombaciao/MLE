import streamlit as st
import numpy as np
import pandas as pd
import plotly_express as px
import yfinance as yf
import datetime
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

st.title('Using Tensorflow Probability for MLE')
st.subheader('First, we compute the mean and sigma using analytical MLE.')
st.subheader('Second, we compute the mean and sigma using TF Probability by optimization and gradients.')

ticker = st.sidebar.text_input('Input Stock Ticker', value='IBM', max_chars=None, key=None, type='default')
st.header(ticker)

start = st.sidebar.date_input('Start Date', datetime.date(2015,5,1))
end = st.sidebar.date_input('End Date', datetime.date(2021,5,21))

df = yf.download(ticker, start=start, end=end, progress=False)

plot = px.line(data_frame=df, x=df.index, y=df['Adj Close'], title=ticker)
st.plotly_chart(plot)

r = tf.math.log(df['Adj Close']/df['Adj Close'].shift(1))
r = r[1:] * 100
plot = px.histogram(x=r, title='Log Daily Returns')
st.plotly_chart(plot)

def mle(x):
    mu = tf.reduce_mean(x)
    sigma = tf.math.sqrt(tf.reduce_mean(tf.math.square(x - mu)))
    return mu, sigma

def nll(dist, x_train):
    return -tf.reduce_mean(dist.log_prob(x_train))

@tf.function
def get_loss_and_grads(dist, x_train):
    with tf.GradientTape() as tape:
        tape.watch(dist.trainable_variables)
        loss = nll(dist, x_train)
        grads = tape.gradient(loss, dist.trainable_variables)
    return loss, grads

optimizer = tf.keras.optimizers.SGD(learning_rate=0.015)

def mle_fit(x):
    t = tfd.Normal(loc=tf.Variable(0.0, name='loc',dtype=tf.float64), scale=tf.Variable(1, name='scale',dtype=tf.float64))
    
    epochs = 1000
    nll_loss = []
    for _ in range(epochs):
        loss, grads = get_loss_and_grads(t, r)
        optimizer.apply_gradients(zip(grads, t.trainable_variables))
        nll_loss.append(loss)
    
    plot = px.line(nll_loss, title='Loss Optimization')
    st.plotly_chart(plot)

    return t.loc, t.scale
    
if(st.sidebar.button("Press Button to Compute MLE")):
    mu, sigma = mle(r)
    mu_fit, sigma_fit = mle_fit(r)
    st.sidebar.write('Mean of Log Daily Return')
    st.sidebar.info(f'{mu.numpy():.5f}')
    st.sidebar.info(f'{mu_fit.numpy():.5f}')
    
    st.sidebar.write('Standard Deviation of Log Daily Return')
    st.sidebar.info(f'{sigma.numpy():.5f}')
    st.sidebar.info(f'{sigma_fit.numpy():.5f}')
    