import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

def plot_history(history, metric_names=["categorical_accuracy", "loss"], filename = "", figsize=(20, 7)):
    """
    plot and saves history from model history
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax = ax.ravel()
    historyDataFrame = pd.DataFrame()
    metrics = metric_names
    for i, met in enumerate(metrics):
        historyDataFrame[met] = history.history[met]
        historyDataFrame["val_" + met] = history.history["val_" + met]
        ax[i].plot(history.history[met])
        ax[i].plot(history.history["val_" + met])
        ax[i].set_title("Model {}".format(met))
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(met)
        ax[i].legend(["train", "val"])

    historyDataFrame.to_csv(filename)
    
    
def plot_pretty_history(file_path, save=True, filename="hist_L_.svg"):
    """
    plot history from saved history df
    """
    df = pd.read_csv(file_path, index_col=0)
    fig, ax = plt.subplots(1,2,figsize=(20,8))
    df[["loss", "val_loss"]].plot(ax=ax[0],
                                  xlabel="epoch",
                                  title="Loss function",
                                  style=["-", "--"],
                                  color=["darkblue", "deepskyblue"])
    df[["categorical_accuracy", "val_categorical_accuracy"]].plot(ax=ax[1],
                                                                  xlabel="epoch",
                                                                  title="Categorical Accuracy function",
                                                                  style=["-", "--"],
                                                                  color=["darkred", "lightcoral"])
    if save:
        fig.savefig(filename, bbox_inches='tight')


def plot_predictions(data, plot_lines=False, filename="estimated_results.svg", cmaps=None, figsize=(13,9), plot_critic=True):
    
    df2 = data.drop(["std_low", "std_high"], axis=1)
    if cmaps==None:
#         blues = sns.color_palette("Blues")
        blues = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, light=0.7)
        reds = sns.dark_palette("red", reverse=True, as_cmap=True)
    else:
        blues = cmaps[0]
        reds=cmaps[1]

    fig, ax = plt.subplots(figsize=figsize)
    if not plot_lines:
    
        ax = sns.scatterplot(x="Temperature", y="P_low",
        data=df2.drop("P_high", axis=1),
        ax=ax, palette=blues, style="L", hue="L")
        
        sns.scatterplot(x="Temperature", y="P_high",
        data=df2.drop("P_low", axis=1), ax=ax,
        legend=False, palette=reds, style="L", hue="L", alpha=0.7)
        
        if plot_critic:
            ax.axvline(x=2/np.log(1+np.sqrt(2)), ymin=0, ymax=1, color='gray', linewidth=5, alpha=0.4)
            
    else:
        uniqueL = pd.unique(df2.L).count()
        dashes=[(1,1) for i in range(uniqueL)]
        ax = sns.lineplot(x="Temperature", y="P_low",
        data=df2.drop(["P_high"], axis=1),
        hue="L", ax=ax, palette=blues, style="L",
        markers=True, dashes=dashes)
        
        sns.lineplot(x="Temperature", y="P_high",
        data=df2.drop(["P_low"], axis=1), hue="L",
        ax=ax, legend=False, palette=reds, style="L",
        markers=True, dashes=dashes, alpha=0.6)
        
        if plot_critic:
            ax.axvline(x=2.269, ymin=0, ymax=1, color='gray', linewidth=5, alpha=0.4)
    
    ax.set_xlim([0.5, 4.0])     
    ax.set_ylabel("Output layer")
    
    if filename:
        fig.savefig(filename, bbox_inches='tight')
    
    del df2
    
def plot_collapsed_predictions(data, filename="scaling1_results.svg", figsize=(13,9)):
    
    df2 = data.drop(["std_low", "std_high"], axis=1)
    # finite size scalling
    v = 1
    Tc = 2/np.log(1+np.sqrt(2))

    df2["t"] = (df2["Temperature"]-Tc)/Tc
    df2["scaling"] = df2["t"]*df2["L"]**(1/v)

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(x="scaling", y="P_low", data=df2[["P_low", "L", "scaling"]],
        hue="L", ax=ax, legend=False, palette="Blues", style="L")
    
    sns.scatterplot(x="scaling", y="P_high", data=df2[["P_high", "L", "scaling"]],
        hue="L", ax=ax, legend=False, palette="Reds", style="L")
        
    ax.set_xlabel(r"$\left ( 1-\frac{T^*}{T_c ^*} \right ) L^{1/\nu}$")
    ax.set_ylabel("Output layer")
    ax.set_xlim([-10,10])
    if filename:
        fig.savefig(filename, bbox_inches='tight')
    
    del df2
    
def plot_finite_size_scaling(intersect_data, plot_critic=True, fit_line=True, lin_regress_data=None, show_errorbars=True, errorbars=None, figsize=(12,9), filename="scaling2_results.svg"):
    Tc = 2/np.log(1+np.sqrt(2))
    fig, ax = plt.subplots(figsize=figsize)
    if not show_errorbars:
        ax.plot(intersect_data["inv_L"], intersect_data["Tc"], "-.^", color="darkblue", label="Finite-size scaling")
    else:
        if not errorbars:
            ax.errorbar(intersect_data["inv_L"], intersect_data["Tc"],
                        yerr=np.std(intersect_data["Tc"]),
                        elinewidth=1, color="darkblue",
                        linestyle="-.", marker="^",
                        label="Finite size scaling")
        else:
            ax.errorbar(intersect_data["inv_L"], intersect_data["Tc"],
                        yerr=errorbars,
                        elinewidth=1, color="darkblue",
                        linestyle="-.", marker="^",
                        label="Finite size scaling result")
    if plot_critic:
        ax.axhline(y=Tc, xmin=0, xmax=1, color='red', linewidth=5, alpha=0.4, label="Tc")
    
    if fit_line and lin_regress_data:
        x_extended, y_fit = lin_regress_data
        ax.plot(x_extended, y_fit, "--r", label="Linear regression")
    
    ax.set_ylabel(r"$T^*/J$")
    ax.set_xlabel("1/L")
#     ax.set_xlim([-0.001, 0.101])
    ax.legend()
    if filename:
        fig.savefig(filename, bbox_inches='tight')
    
