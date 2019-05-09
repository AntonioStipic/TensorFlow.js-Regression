
let xs = [];
let ys = [];

let a, b, c, d, e;

const learningRate = 1;
const optimizer = tf.train.adamax(learningRate);

function setup() {
    createCanvas(640, 480);
    background(0);
    stroke(255);

    a = tf.scalar(random(-1, 1)).variable();
    b = tf.scalar(random(-1, 1)).variable();
    c = tf.scalar(random(-1, 1)).variable();
    d = tf.scalar(random(-1, 1)).variable();
    e = tf.scalar(random(-1, 1)).variable();
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
        let px = map(xs[i], -1, 1, 0, width);
        let py = map(ys[i], -1, 1, height, 0);
        point(px, py);
    }
    strokeWeight(2);

    curveX = [];
    for (let x = -1; x <= 1; x += 0.02) {
        curveX.push(x);
    }

    let y = tf.tidy(() => predict(curveX));
    let curveY = y.dataSync();
    y.dispose();

    beginShape();
    noFill();
    stroke(255);
    strokeWeight(2);
    for (let i = 0; i <= curveX.length; i++) {
        let x = map(curveX[i], -1, 1, 0, width);
        let y = map(curveY[i], -1, 1, height, 0);
        vertex(x, y);
    }
    endShape();

    //console.log(tf.memory().numTensors);

    document.getElementById("formula").innerHTML = "y = " + a.dataSync()[0].toFixed(2) + "x^4 + " + b.dataSync()[0].toFixed(2) + "x^3 + " + c.dataSync()[0].toFixed(2) + "x^2 + " + d.dataSync()[0].toFixed(2) + "x + " + e.dataSync()[0].toFixed(2);
}

function mouseClicked() {

    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);

    if (x <= 1 && x >= -1 && y <= 1 && y >= -1) {
        xs.push(x);
        ys.push(y);
    }
}

function predict(xs) {
    const tfxs = tf.tensor1d(xs);

    // y = ax^3 + bx^2 + cx + d;
    const tfys = tfxs.pow(tf.scalar(4)).mul(a).add(tfxs.pow(tf.scalar(3)).mul(b)).add(tfxs.pow(tf.scalar(2)).mul(c)).add(tfxs.mul(d)).add(e);
    return tfys;
}

function loss(pred, labels) {
    return pred.sub(labels).square().mean();
}