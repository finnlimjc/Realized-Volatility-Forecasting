import matplotlib.pyplot as plt

def plot_volatility_estimate(pred, actual, title) -> plt.Figure:
    plt.plot(pred, color='red', label='predicted')
    plt.plot(actual, color='black', alpha=0.5, label='actual')
    plt.title(title)
    plt.ylabel("Volatility")
    plt.xlabel("Dates")
    plt.legend()
    fig = plt.gcf()
    return fig