#include <iostream>
#include <vector>
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
    const vector<double>& input
) {
    vector<double> output(W.size());

    for (size_t j = 0; j < W.size(); j++) {
        double z = dot_product(W[j], input) + b[j];
        output[j] = relu(z);
    }
    return output;
}

/* Full network forward */
vector<double> feed_forward(
    const vector<double>& x,
    const vector<vector<double>>& W1,
    const vector<double>& B1,
    const vector<vector<double>>& W2,
    const vector<double>& B2
) {
    vector<double> hidden = feed_forward_layer(W1, B1, x);
    vector<double> output = feed_forward_layer(W2, B2, hidden);
    return output;
}

int main() {
    vector<double> x = {1.0, 0.5};

    vector<vector<double>> W1 = {
        {0.2, -0.1},
        {0.4,  0.3}
    };
    vector<double> B1 = {0.1, -0.2};

    vector<vector<double>> W2 = {
        {0.7, -0.5}
    };
    vector<double> B2 = {0.0};

    vector<double> y = feed_forward(x, W1, B1, W2, B2);

    cout << "Output: " << y[0] << endl;
    return 0;
}
