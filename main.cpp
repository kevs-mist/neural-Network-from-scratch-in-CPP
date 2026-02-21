#include <iostream>
#include <vector>
#include <cmath>
#include <random>
using namespace std;

/* Activation */
double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

/* Dot product */
double dot_product(const vector<double>& x,
                   const vector<double>& y) {
    double res = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        res += x[i] * y[i];
    }
    return res;
}

/* One layer forward */
vector<double> feed_forward_layer(
    const vector<vector<double>>& W,
    const vector<double>& b,
    const vector<double>& input,
    bool apply_relu = true
) {
    vector<double> output(W.size());
    for (size_t j = 0; j < W.size(); j++) {
        double z = dot_product(W[j], input) + b[j];
        output[j] = apply_relu ? relu(z) : z;
    }
    return output;
}

/* Softmax */
vector<double> softmax(const vector<double>& z) {
    vector<double> probs(z.size());
    double sum = 0.0;

    for (double v : z)
        sum += exp(v);

    for (size_t i = 0; i < z.size(); i++)
        probs[i] = exp(z[i]) / sum;

    return probs;
}

/* Cross-entropy loss */
double cross_entropy(
    const vector<double>& y,
    const vector<double>& y_hat
) {
    double loss = 0.0;
    for (size_t i = 0; i < y.size(); i++)
        loss -= y[i] * log(y_hat[i]);
    return loss;
}


void generate_dataset(
    vector<vector<double>>& X,
    vector<vector<double>>& Y,
    int samples
) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(-1.0, 1.0);

    for (int i = 0; i < samples; i++) {
        double x1 = dist(gen);
        double x2 = dist(gen);

        vector<double> x = {x1, x2};

        double score = x1 + 2 * x2;
        vector<double> y = {score > 0 ? 1.0 : 0.0};

        X.push_back(x);
        Y.push_back(y);
    }
}

double accuracy(
    const vector<vector<double>>& X,
    const vector<vector<double>>& Y,
    const vector<vector<double>>& W1,
    const vector<double>& B1,
    const vector<vector<double>>& W2,
    const vector<double>& B2
) {
    int correct = 0;

    for (size_t i = 0; i < X.size(); i++) {
        vector<double> hidden = feed_forward_layer(W1, B1, X[i]);
        vector<double> logits = feed_forward_layer(W2, B2, hidden, false);
        vector<double> y_hat = softmax(logits);

        int pred = y_hat[0] >= 0.5 ? 1 : 0;
        int true_label = Y[i][0];

        if (pred == true_label)
            correct++;
    }

    return double(correct) / X.size();
}

int main() {

        vector<vector<double>> X, Y;
        generate_dataset(X, Y, 10000);

        // 80 / 20 split
        int split = 8000;

        vector<vector<double>> X_train(X.begin(), X.begin() + split);
        vector<vector<double>> Y_train(Y.begin(), Y.begin() + split);

        vector<vector<double>> X_test(X.begin() + split, X.end());
        vector<vector<double>> Y_test(Y.begin() + split, Y.end());

        vector<vector<double>> W1 = {
            {0.2, -0.1},
            {0.4,  0.3}
        };
        vector<double> B1 = {0.1, -0.2};

        vector<vector<double>> W2 = {
            {0.7, -0.5}
        };
        vector<double> B2 = {0.0};

        double train_loss = 0.0;

        for (size_t i = 0; i < X_train.size(); i++) {
            vector<double> hidden = feed_forward_layer(W1, B1, X_train[i]);
            vector<double> logits = feed_forward_layer(W2, B2, hidden, false);
            vector<double> y_hat = softmax(logits);

            train_loss += cross_entropy(Y_train[i], y_hat);
        }

        train_loss /= X_train.size();

        double test_acc = accuracy(X_test, Y_test, W1, B1, W2, B2);

        cout << "Train samples: " << X_train.size() << endl;
        cout << "Test samples:  " << X_test.size() << endl;
        cout << "Train loss:    " << train_loss << endl;
        cout << "Test accuracy: " << test_acc * 100 << "%" << endl;

        return 0;
    }
