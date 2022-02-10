import tensorflow as tf, matplotlib.pyplot as plt, os, numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(77)

x_train_data = [1,2,3]
y_train_data = [1,2,3]
x_test_data = [4,5,6]
y_test_data = [4,5,6]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])


w = tf.compat.v1.Variable(0, dtype=tf.float32)
# w = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)

hypothesis = x * w

# loss = tf.keras.losses.MeanAbsoluteError(hypothesis,y)  # MAE 어떻게 써야하나..
loss = tf.reduce_mean(tf.square(hypothesis - y))          # MSE

lr = 0.21
gradient = tf.reduce_mean(( x * w - y) * x)
descent = w - lr * gradient
update = w.assign(descent)      # w에 descent를 넣어주겠다. -> w = w - lr * gradient 

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

step = 0
print(f"Step\tdescent_val\tgradient_val\tloss_val\tw_val")
while True:
    step += 1
    
    # 내방식
    _,descent_val,gradient_val,loss_val,w_val = sess.run([update,descent,gradient,loss,w], feed_dict={x:x_train_data,y:y_train_data})
    print(f"{step:04d}\t{descent_val:.7f} \t{gradient_val:.7f} \t{loss_val:.10f} \t{w_val:.7f}")

    if loss_val <= 1e-10: 
        predict = x_test*w_val
        
        y_predict = sess.run(predict,feed_dict={x_test:x_test_data})
        
        r2 = r2_score(y_test_data,y_predict)
        mae = mean_absolute_error(y_test_data,y_predict)
        
        print(f"r2스코어는 : {r2}")
        print(f"MAE는 : {mae}")
        
        break
sess.close()



