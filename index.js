const model = tf.sequential();
model.add(tf.layers.dense({inputShape:[1], units: 1}));
model.compile({
  loss: 'meanSquaredError',
  optimizer: 'sgd',
  metrics: ['accuracy']
});

train_data = async () => {
  const data = tf.tensor2d([1,2,3,4,5,6,7,8], [8, 1]);
  const labels = tf.tensor2d([-1,-2,3,-4,-5,-6,-7,-8], [8, 1]);

  await model.fit(data, labels, {epochs: 500});
  alert('Data Training Completed');
  document.getElementById('input').disabled = false
  document.getElementById('go').disabled = false
}

train_data();

predictOutput = (input) => {
  document.getElementById('result').innerText = model.predict(tf.tensor2d([parseInt(input)], [1,1]));
};