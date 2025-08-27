# Sigmoid activation function
sigmoid <- function(x) {
    1 / (1 + exp(-x))
}

softmax <- function(x) {
    exp_x <- exp(x - max(x))
    exp_x / sum(exp_x)
}

sigmoid(1)
sigmoid(-100)

x <- c(1, 2, 3)
