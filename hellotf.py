import tensorflow as tf


#TF constants
hello_constant = tf.constant('Hello World')
A = tf.constant(1234)
AA = tf.constant(1111)
AAA = tf.add(A, AA)
B = tf.constant([123,456,789])
C = tf.constant([[123,456,789], [222,333,444]])
D = tf.add(B, C)
with tf.Session() as sess:
    output = sess.run(hello_constant)
    print(output)
    numoutput = sess.run(AAA)
    print (numoutput)
    matoutput = sess.run(D)
    print(D)

#TF placeholders
#allows for setting input right before the session runs
#values provided as a dictionary to the session run method

x = tf.placeholder(tf.string) #setting datatype but not value
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.int32)
sum = y+z
with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello dictionary value'})
    sumoutput = sess.run(sum, feed_dict={y:4, z:9})
    print (output)
    print(sumoutput)


#TF math is different from regular math
tfsum = tf.add(y,z)
with tf.Session() as sess:
    tfoutput = sess.run(tfsum, feed_dict={z: 55, y: 100})
    print(tfoutput)

#TF type conversion can be necessary
tfint = tf.constant(4)
tffloat = tf.constant(55.5)
tfprod = tf.multiply(tf.cast(tfint, tf.float32), tffloat)
with tf.Session() as sess:
    tfoutput = sess.run(tfprod)
    print(tfoutput)
