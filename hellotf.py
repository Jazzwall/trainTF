import tensorflow as tf

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