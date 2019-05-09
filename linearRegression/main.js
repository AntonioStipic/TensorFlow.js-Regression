
let xs = [];
let ys = [];

// Variables k and l for y = kx + l;
let k, l;

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function setup() {
    createCanvas(640, 480);
    background(0);
    stroke(255);

    k = tf.scalar(random(1)).variable();
    l = tf.scalar(random(1)).variable();
}

function draw() {

    if (xs.length > 0) {
        tf.tidy(() => {
            const tfys = tf.tensor1d(ys);
            optimizer.minimize(() => loss(predict(xs), tfys));
        });
    }

    background(0);
    strokeWeight(10);
    for (let i = 0; i < xs.length; i++) {
        let px = map(xs[i], 0, 1, 0, width);
        let py = map(ys[i], 0, 1, height, 0);
        point(px, py);
    }
    strokeWeight(2);

    let y = tf.tidy(() => predict([0, 1]));
    let lineY = y.dataSync();
    y.dispose();

    let x1 = map(0, 0, 1, 0, 640);
    let x2 = map(1, 0, 1, 0, 640);

    let y1 = map(lineY[0], 0, 1, height, 0);
    let y2 = map(lineY[1], 0, 1, height, 0);

    line(x1, y1, x2, y2);

    //console.log(tf.memory().numTensors);
}

function mouseClicked() {

    let x = map(mouseX, 0, width, 0, 1);
    let y = map(mouseY, 0, height, 1, 0);

    xs.push(x);
    ys.push(y);
}

function predict(xs) {
    const tfxs = tf.tensor1d(xs);

    // y = kx + l;
    const tfys = tfxs.mul(k).add(l);
    return tfys;
}

function loss(pred, labels) {
    return pred.sub(labels).square().mean();
}