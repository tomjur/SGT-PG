from ompl import base as ob
from ompl import geometric as og
import tensorflow as tf

space = ob.SE2StateSpace()
# set lower and upper bounds
bounds = ob.RealVectorBounds(2)
bounds.setLow(-55)
bounds.setHigh(55)
space.setBounds(bounds)
og.SimpleSetup(space)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


print "done"