import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
with tf.compat.v1.Session() as sess:  # 텐서플로우 세션 연결 수립(->시작)

   a = tf.compat.v1.constant(120, name="a")
   b = tf.compat.v1.constant(130, name="b")
   c = tf.compat.v1.constant(140, name="c")   # 텐서 세팅(상수)
   v = tf.compat.v1.Variable(0, name="v") #텐서 세팅(변수)

   calc_op = a + b + c  # 노드 구조(연산식) 구축-
   assign_op = tf.compat.v1.assign(v, calc_op)  # 계산 결과와 변수 연결

   sess.run(assign_op)  # 연산 실행 (run())
   print(sess.run(v))  # v에 담긴 값을 로드해서 출력
