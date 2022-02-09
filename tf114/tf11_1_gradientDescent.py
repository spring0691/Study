import tensorflow as tf, matplotlib.pyplot as plt, os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(77)

x_train = [1,2,3]
y_train = [1,2,3]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable(-10, dtype=tf.float32)
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

w_history = []
loss_history = []

step = 0
print(f"Step\tdescent_val\tgradient_val\tloss_val\tw_val")
while True:
    step += 1
    # 쌤방식
    # w_val = sess.run(update, feed_dict={x:x_train,y:y_train})
    # print(step+1,'\t',sess.run(loss,feed_dict={x:x_train,y:y_train}), sess.run(w))  # 여기서 sess.run이 한번 더 실행되서 값이 밀린다.
    
    # 내방식
    _,descent_val,gradient_val,loss_val,w_val = sess.run([update,descent,gradient,loss,w], feed_dict={x:x_train,y:y_train})
    print(f"{step:04d}\t{descent_val:.7f} \t{gradient_val:.7f} \t{loss_val:.10f} \t{w_val:.7f}")
    
    w_history.append(w_val)
    loss_history.append(loss_val)

    if loss_val <= 1e-10: break
sess.close()

'''
print("================ w_history =================")
print(w_history)
print("==============loss_history =================")
print(loss_history)

plt.plot(w_history,loss_history)
plt.xlabel('Weight')
plt.ylabel('loss')
plt.show()
'''