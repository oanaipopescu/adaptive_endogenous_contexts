
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib import cm
import matplotlib.transforms as mtrans
from config_generator import generate_configurations, generate_string_from_params, generate_name_from_params, load_configurations

import numpy as np

def get_plotting_params():
    plotting_params = {
        'legend.fontsize': 16,
        'lines.linewidth': 2.5,
        'lines.markersize': 6,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.labelsize': 14,
        'font.size': 16,
        'title.size': 16,
        'figsize_regimes': (6, 8),  # Adjusted for better readability
        'figsize_union': (8, 6),    # Adjusted for better readability
        'common_fig_width': 8,      # Common width for side-by-side plots
        'common_fig_height': 6      # Common height for side-by-side plots
    }
    # Define clean metric names and method names
    metric_clean_names = {
        'union_tpr': 'TPR', 'union_fpr': 'FPR', 'adj_recall': 'adj-Recall', 'union_adj_prec': 'adj-Precision',
        'union_edge_prec': 'Union Edgemark Prec.', 'union_edge_rec': 'Union Edgemark Rec.',
        'tpr': 'TPR', 'fpr': 'FPR', 'adj_rec': 'Recall', 'adj_prec': 'Precision',
        'edge_prec': 'Edge Precision', 'edge_rec': 'Edge Recall',
        'equal_regimes': '# equal regimes', 'reg_fpr': 'Reg.ind. FPR', 'reg_tpr': 'Reg.ind. TPR',
        'f1': 'F1', 'union_f1': 'F1', 'intersection': 'Intersection'
    }
    method_names = {
        '_ymask': 'PC-M', '_regimes': 'PC-AR', '_pcmci': 'PC-P', '_intersection': 'PC-B', '_fci': 'FCI', '_nod': 'CD-NOD'
    }
    colors = {
        '_intersection': '#1F77B4',
        '_ymask': '#D62728',
        '_regimes': '#2CA02C', 
        '_pcmci': '#FF7F0E',
        '_fci': '#A2AEBB',
        '_nod': '#1C3144'
    }

    linestyles = {'False': '-', 'True': '--', 'and_parents': ':'}
    
    return plotting_params, metric_clean_names, method_names, colors, linestyles


def load_metrics_regime(metric, method, regime, sample_sizes, metrics_folder, save_folder, config, regime_children_value):
    metric_pp_list, metric_errors_list = [], []
    failed_files = []
    for sample_size in sample_sizes:
        metrics_file_path = get_metrics_file_path(metrics_folder, save_folder, config, sample_size, regime_children_value, 'regimes')
        try:
            with open(metrics_file_path, 'rb') as file:
                metrics_dict = pickle.load(file, encoding='latin1')
            metrics_key = metric + method
            if metrics_key in metrics_dict[regime]:
                metric_pp_list.append(metrics_dict[regime][metrics_key][0])
                metric_errors_list.append(metrics_dict[regime][metrics_key][1])
        except FileNotFoundError:
            metric_pp_list.append(0.)
            metric_errors_list.append(0.)
            failed_files.append(metrics_file_path)

    return metric_pp_list, metric_errors_list, failed_files

def load_metrics_avg_regimes(metric, method, sample_sizes, metrics_folder, save_folder, config, regime_children_value):
    metric_pp_list, metric_errors_list = [], []
    failed_files = []
    for sample_size in sample_sizes:
        metrics_file_path = get_metrics_file_path(metrics_folder, save_folder, config, sample_size, regime_children_value, 'avg_regimes')
        try:
            with open(metrics_file_path, 'rb') as file:
                metrics_dict = pickle.load(file, encoding='latin1')
            metrics_key = metric + method
            if metrics_key in metrics_dict:
                metric_pp_list.append(metrics_dict[metrics_key][0])
                metric_errors_list.append(metrics_dict[metrics_key][1])
        except FileNotFoundError:
            metric_pp_list.append(0.)
            metric_errors_list.append(0.)
            failed_files.append(metrics_file_path)

    return metric_pp_list, metric_errors_list, failed_files
    
def generate_time_conv_plot_regimes(all_configurations, sample_sizes=[10, 20, 30, 50, 100, 150, 200], regime_children_known=None, metrics_folder=None, combined_plot=False, axs=None, plot_avg='False', metrics_list=['tpr', 'fpr'], show_xticks=False):
    plotting_params, metric_clean_names, method_names, colors, linestyles = get_plotting_params()
    config = all_configurations[-1][0]
    N, density, max_lag, pc_alpha, _, reg_v, nb_changed_links, nb_regimes, nb_repeats, cycles_only, remove_only, use_cmiknnmixed, imbalance_factor, save_folder = config

    if combined_plot and axs is None:
        raise ValueError("Axes must be provided when plotting in combined mode.")

    
    if not combined_plot:
        fig_width = plotting_params['common_fig_width'] // 2
        fig_height = plotting_params['common_fig_height']
        fig, axs = plt.subplots(nb_regimes, len(metrics_list), figsize=(fig_width, fig_height))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        axs = axs.ravel()
        fig.suptitle(f"Contexts for Nodes: {N}, Contexts: {nb_regimes}, # Changed Links: {nb_changed_links}, Density: {density}, Cycles: {cycles_only}, Remove Only: {remove_only}", fontsize=plotting_params['title.size'])

    if plot_avg == 'False':
        for j, (regime, metric) in enumerate([(regime, metric) for regime in range(nb_regimes) for metric in metrics_list]):
            if len(metrics_list) > 1:
                ax = axs[j]
            else: 
                ax = axs
            for regime_children_value in regime_children_known:
                for method in ['_regimes', '_ymask', '_intersection']:
                    metric_pp_list, metric_errors_list, failed_files = load_metrics_regime(metric, method, regime, sample_sizes, metrics_folder, save_folder, config, regime_children_value)
                    ax.plot(sample_sizes[:len(metric_pp_list)], metric_pp_list, 'o-', markersize=plotting_params['lines.markersize'], label=f"{method_names[method]} adj. {get_regime_label(regime_children_value)}", linestyle=linestyles.get(str(regime_children_value)), color=colors.get(method))
                    ax.errorbar(sample_sizes[:len(metric_pp_list)], metric_pp_list, yerr=metric_errors_list, color=colors.get(method), linestyle=linestyles.get(str(regime_children_value)))   
            if show_xticks:
                ax.set_xticks(sample_sizes)
                ax.set_xticklabels(sample_sizes, rotation=40, ha='right', fontsize=plotting_params['xtick.labelsize'])
            
            ylabel = 'Regime ' + str(regime)
            # + ' ' + metric_clean_names[metric] if metric == "tpr" else metric_clean_names[metric]
            if 'tpr' in metric:
                ax.set_ylabel(ylabel, fontsize=plotting_params['axes.labelsize'])
            if not combined_plot:
                ax.set_xlabel('Sample Size', fontsize=plotting_params['axes.labelsize'])
            else:
                if regime == nb_regimes - 1:
                    ax.set_xlabel('Sample Size', fontsize=plotting_params['title.size'])

            if 'tpr' in metric:
                xticks = [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
                ax.set_yticks(xticks) 
                ax.set_yticklabels([f"{i:.2f}" for i in xticks], 
                               rotation=0, ha='right', fontsize=plotting_params['ytick.labelsize'])
            elif 'fpr' in metric:
                yticks = [0., 0.01, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
                ax.set_yticks(yticks)  # Assuming you want 6 ticks from 0 to 1
                ax.set_yticklabels([f"{i:.2f}" for i in yticks], 
                                   rotation=0, ha='right', fontsize=plotting_params['ytick.labelsize'])
                ax.set_ylim([0, 0.1])

            ax.grid(color='lightgray')

            if 'fpr' in metric:
                ax.hlines(y=0.05, xmin=sample_sizes[0], xmax=sample_sizes[-1], colors='black', linestyles='-', lw=1)
                
        if not combined_plot:
            ax.set_xlabel('Sample Size', fontsize=plotting_params['title.size'])
            ax.set_title(metric_clean_names[metric], fontsize=plotting_params['title.size'])
            fig.savefig(os.path.join(figure_path, file_name + '_regimes' + '_'.join(metric for metric in metrics_list)  + '_' + '_'.join(known for known in regime_children_known) + '.png'), bbox_inches='tight')
            plt.close(fig)
    else:
        for j, metric in enumerate(metrics_list):
            if len(metrics_list) > 1:
                ax = axs[j]
            else:
                ax = axs[j]
            for regime_children_value in regime_children_known:
                for method in ['_regimes', '_ymask', '_intersection']:
                    metric_pp_list, metric_errors_list, failed_files = load_metrics_avg_regimes(metric, method, sample_sizes, metrics_folder, save_folder, config, regime_children_value)
                    ax.plot(sample_sizes[:len(metric_pp_list)], metric_pp_list, 'o-', markersize=plotting_params['lines.markersize'], label=f"{method_names[method]} adj. {get_regime_label(regime_children_value)}", linestyle=linestyles.get(str(regime_children_value)), color=colors.get(method))
                    ax.errorbar(sample_sizes[:len(metric_pp_list)], metric_pp_list, yerr=metric_errors_list, color=colors.get(method), linestyle=linestyles.get(str(regime_children_value)))
            if show_xticks:
                ax.set_xticks(sample_sizes)
                ax.set_xticklabels(sample_sizes, rotation=40, ha='right', fontsize=plotting_params['xtick.labelsize'])

            if 'tpr' in metric:
                xticks = [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
                ax.set_yticks(xticks) 
                ax.set_yticklabels([f"{i:.2f}" for i in xticks], 
                               rotation=0, ha='right', fontsize=plotting_params['ytick.labelsize'])
            elif 'fpr' in metric:
                yticks = [0., 0.01, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
                ax.set_yticks(yticks)  # Assuming you want 6 ticks from 0 to 1
                ax.set_yticklabels([f"{i:.2f}" for i in yticks], 
                                   rotation=0, ha='right', fontsize=plotting_params['ytick.labelsize'])
                ax.set_ylim([0, 0.1])

            ax.grid(color='lightgray')

            if 'fpr' in metric:
                ax.hlines(y=0.05, xmin=sample_sizes[0], xmax=sample_sizes[-1], colors='black', linestyles='-', lw=1)
            
            ylabel = 'Avg. Contexts'
            if j==0:
                ax.set_ylabel(ylabel, fontsize=plotting_params['title.size'])
                
            ax.set_xlabel('Sample Size', fontsize=plotting_params['title.size'])
                    
        if not combined_plot:
            ax.set_xlabel('Sample Size', fontsize=plotting_params['title.size'])
            ax.set_title(metric_clean_names[metric], fontsize=plotting_params['title.size'])
            fig.savefig(os.path.join(figure_path, file_name + '_avg_regimes' + '_'.join(metric for metric in metrics_list)  + '_' + '_'.join(str(known) for known in regime_children_known) + '.png'), bbox_inches='tight')
            plt.close(fig)
    

    return failed_files

def load_metrics_file_union(metric, method, sample_sizes, metrics_folder, save_folder, config, regime_children_value, suffix=None):
    metric_pp_list, metric_errors_list = [], []
    failed_files = []
    for sample_size in sample_sizes:
        metrics_file_path = get_metrics_file_path(metrics_folder, save_folder, config[-1][0], sample_size, regime_children_value, 'union', suffix=suffix)
        try:
            with open(metrics_file_path, 'rb') as file:
                metrics_dict = pickle.load(file, encoding='latin1')
            metric_key = metric + method
            if metric_key in metrics_dict:
                metric_pp_list.append(metrics_dict[metric_key][0])
                metric_errors_list.append(metrics_dict[metric_key][1])
        except FileNotFoundError:
            metric_pp_list.append(0.)
            metric_errors_list.append(0.)
            failed_files.append(metrics_file_path)
    return metric_pp_list, metric_errors_list, failed_files
                

def generate_time_conv_plot_union(all_configurations, sample_sizes=[500, 1000], regime_children_known=None, metrics_folder=None, metrics_folder_fci=None, combined_plot=False, axs=None, plot_avg='False', metrics_list=['tpr', 'fpr'], plot_fci='False', show_xticks=False):
    plotting_params, metric_clean_names, method_names, colors, linestyles = get_plotting_params()

    metrics_list = ['union_' + metric for metric in metrics_list]
    
    N, density, max_lag, pc_alpha, _, reg_v, nb_changed_links, nb_regimes, nb_repeats, cycles_only, remove_only, use_cmiknnmixed, imbalance_factor, save_folder = all_configurations[-1][0]
    
    if combined_plot and axs is None:
        raise ValueError("Axes must be provided when plotting in combined mode.")
    
    fig_save_foldername = '/plot_figures_fci/'
    file_name = generate_name_from_params(all_configurations[-1][0])
    figure_path = metrics_folder + fig_save_foldername if metrics_folder else save_folder + fig_save_foldername
    os.makedirs(figure_path, exist_ok=True)
    
    if not combined_plot:
        cols = len(metrics_list)
        fig_width = plotting_params['common_fig_width'] // 2
        fig_height = plotting_params['common_fig_height']
        fig, axs = plt.subplots(1, cols, figsize=(fig_width, fig_height))
        plt.subplots_adjust(hspace=0.25, wspace=0.25)
        axs = axs.ravel()
        fig.suptitle(f"Labeled Union for Nodes: {N}, Contexts: {nb_regimes}, Changed Links: {nb_changed_links}, Density: {density}, Cycles: {cycles_only}, Remove Only: {remove_only}", fontsize=plotting_params['title.size'])

    all_handles, all_labels = [], []
    
    for j, metric in enumerate(metrics_list):
        if not 'Ax' in str(type(axs)):
            ax = axs[j]
        else:
            ax = axs
        for regime_children_value in regime_children_known:
            # for method in method_names:
            for method in ['_regimes', '_ymask', '_intersection', '_pcmci']:
                metric_pp_list, metric_errors_list, failed_files = load_metrics_file_union(metric, method, sample_sizes, metrics_folder, save_folder, all_configurations, regime_children_value)
                ax.plot(sample_sizes[:len(metric_pp_list)], metric_pp_list, 'o', markersize=plotting_params['lines.markersize'], label=f"{method_names[method]} adj. {get_regime_label(regime_children_value)}", linestyle=linestyles.get(str(regime_children_value)), color=colors.get(method))
                ax.errorbar(sample_sizes[:len(metric_pp_list)], metric_pp_list, yerr=metric_errors_list, color=colors.get(method), linestyle=linestyles.get(str(regime_children_value)))

        if plot_fci == 'True':
            print('plotting fci')
            if regime_children_value == False:
                for method in ['_fci', '_nod']:
                        metric_pp_list, metric_errors_list, failed_files = load_metrics_file_union(metric, method, sample_sizes, metrics_folder_fci, save_folder, all_configurations, regime_children_value, suffix='_fci')
                        ax.plot(sample_sizes[:len(metric_pp_list)], metric_pp_list, 'o', markersize=plotting_params['lines.markersize'], label=f"{method_names[method]} adj. {get_regime_label(regime_children_value)}", linestyle=linestyles.get(str(regime_children_value)), color=colors.get(method))
                        ax.errorbar(sample_sizes[:len(metric_pp_list)], metric_pp_list, yerr=metric_errors_list, color=colors.get(method), linestyle=linestyles.get(str(regime_children_value)))

        if show_xticks:
            ax.set_xticks(sample_sizes)
            ax.set_xticklabels(sample_sizes, rotation=40, ha='right', fontsize=plotting_params['xtick.labelsize'])
        if 'tpr' in metric:
            xticks = [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
            ax.set_yticks(xticks) 
            ax.set_yticklabels([f"{i:.2f}" for i in xticks], 
                               rotation=0, ha='right', fontsize=plotting_params['ytick.labelsize'])
        elif 'fpr' in metric:
            yticks = [0., 0.01, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
            ax.set_yticks(yticks)  # Assuming you want 6 ticks from 0 to 1
            ax.set_yticklabels([f"{i:.2f}" for i in yticks], 
                               rotation=0, ha='right', fontsize=plotting_params['ytick.labelsize'])
            ax.set_ylim([0, 0.1])
            
        
        if metric == 'union_fpr':
            ax.hlines(y=0.05, xmin=min(sample_sizes), xmax=max(sample_sizes), colors='black', linestyles='-', lw=1)
        if j == 0:
            ax.set_ylabel('Union', fontsize=plotting_params['title.size'])
        # + ' ' + metric_clean_names[metric], fontsize=plotting_params['axes.labelsize'])

        ax.set_title(metric_clean_names[metric], fontsize=plotting_params['title.size'])
        ax.grid(color='lightgray')
        
    if not combined_plot:
        ax.set_xlabel('Sample Size', fontsize=plotting_params['title.size'])
        handles, labels = axs[0].get_legend_handles_labels()
        save_legend(fig, handles, labels, plotting_params, figure_path, 'legend_union' + '_'.join(str(known) for known in regime_children_known) + '.png')
        fig.savefig(figure_path + file_name + '_union' + '_'.join(metric in metrics_list)  + '_' + '_'.join(str(known) for known in regime_children_known) + '.png', bbox_inches='tight')
        plt.close(fig)

    return failed_files

def save_legend(fig, handles, labels, plotting_params, figure_path, legend_filename):
    fig_legend = plt.figure(figsize=(3, 3))
    fig_legend.legend(handles, labels, loc='center', fontsize=plotting_params['legend.fontsize'])
    fig_legend.savefig(os.path.join(figure_path, legend_filename), bbox_inches='tight')
    plt.close(fig_legend)

def get_metrics_file_path(metrics_folder, save_folder, config, sample_size, regime_children_value, prefix, suffix=None):
    para_setup = (config[0], config[1], config[2], config[3], sample_size, regime_children_value, config[6], config[7], config[8], config[9], config[10], config[11], config[12], config[13])
    file_name = generate_name_from_params(para_setup)
    metrics_result_path = os.path.join(metrics_folder if metrics_folder else save_folder)
    if suffix is None:
        suf = ''
    else:
        suf = suffix
    metrics_file_path = os.path.join(metrics_result_path, file_name + f'_{prefix}{suf}.dat')
    return metrics_file_path


def get_regime_label(value):
    if value == 'and_parents':
        return '$R$-all'
    elif value == True:
        return '$R$-ch.'
    else:
        return 'none'

def generate_fci_plots(all_configurations, sample_sizes, regime_children_known, metrics_folder=None, metrics_folder_fci=None, metrics_list=['tpr', 'fpr'], plot_avg='False', plot_fci='False'):
    plotting_params, metric_clean_names, method_names, colors, linestyles = get_plotting_params()
    
    config = all_configurations[-1][0]
    N, density, max_lag, pc_alpha, _, _, nb_changed_links, nb_regimes, _, cycles_only, remove_only, _, _, save_folder = config
    
    fig_save_foldername = 'plot_figures_fci/'
    figure_path = os.path.join(metrics_folder, fig_save_foldername)
    print(figure_path)
    os.makedirs(figure_path, exist_ok=True)
    
    file_name = generate_name_from_params(config)
    if len(metrics_list) == 1:
        fig, axs = plt.subplots(1, len(metrics_list), figsize=(plotting_params['common_fig_width'] // 2, plotting_params['common_fig_height'] // 2 + 2))
    else:
        fig, axs = plt.subplots(1, len(metrics_list), figsize=(plotting_params['common_fig_width'], plotting_params['common_fig_height'] // 2 + 1))
        plt.subplots_adjust(hspace=0.25, wspace=0.2, top=0.79)

    if remove_only == True:
        ops = 'Remove'
    else:
        ops = 'Remove, Add, Flip'
    fig.suptitle(f"Nodes: {N}, Contexts: {nb_regimes}, Changed Links: {nb_changed_links}, Density: {density},\nCycles: {cycles_only}, Op.: {ops}", fontsize=plotting_params['title.size'])
    
    # Plot Union Metrics
    generate_time_conv_plot_union(
        all_configurations, sample_sizes=sample_sizes, regime_children_known=regime_children_known, 
        metrics_folder=metrics_folder, metrics_folder_fci=metrics_folder_fci, combined_plot=True, axs=axs, metrics_list=metrics_list, plot_avg=plot_avg, plot_fci=plot_fci
    )
    
    ax_for_handles = axs[0]
    handles, labels = ax_for_handles.get_legend_handles_labels()

    fig_legend = plt.figure(figsize=(4, 3))
    fig_legend.legend(handles, labels, loc='lower center', ncol=4,  bbox_to_anchor=(0.5, -0.05), fontsize=plotting_params['legend.fontsize'])
    # fig_legend.legend(handles, labels, loc='center', ncol=1, fontsize=plotting_params['legend.fontsize'])
    plt.tight_layout()
    # Save the legend as a separate file
    fig_legend.savefig(os.path.join(figure_path, 'legend_fci' + '_'.join(str(known) for known in regime_children_known) + '.png'), bbox_inches='tight')
    plt.close(fig_legend)
    
    plt.savefig(os.path.join(figure_path, file_name + '_fci_' + '_'.join(metric for metric in metrics_list) + '_' + '_'.join(str(known) for known in regime_children_known) + '.png'), bbox_inches='tight')
    plt.close(fig)

def generate_combined_plots(all_configurations, sample_sizes, regime_children_known, metrics_folder=None, metrics_folder_fci=None, metrics_list=['tpr', 'fpr'], plot_avg='False', plot_fci='False'):
    plotting_params, metric_clean_names, method_names, colors, linestyles = get_plotting_params()
    
    config = all_configurations[-1][0]
    N, density, max_lag, pc_alpha, _, _, nb_changed_links, nb_regimes, _, cycles_only, remove_only, _, _, save_folder = config
    
    fig_save_foldername = 'plot_figures_fci/'
    figure_path = os.path.join(metrics_folder, fig_save_foldername)
    print(figure_path)
    os.makedirs(figure_path, exist_ok=True)
    
    file_name = generate_name_from_params(config)
        
    if plot_avg == 'False':
        fig, axs = plt.subplots(1 + nb_regimes, len(metrics_list), figsize=(plotting_params['common_fig_width'], plotting_params['common_fig_height'] + 2 * nb_regimes))
        plt.subplots_adjust(hspace=0.25, wspace=0.2, top=0.92)
    else:
        if len(metrics_list) == 1:
            fig, axs = plt.subplots(2, len(metrics_list), figsize=(plotting_params['common_fig_width'] // 2, plotting_params['common_fig_height'] // 2 + 2))
        else:
            fig, axs = plt.subplots(2, len(metrics_list), figsize=(plotting_params['common_fig_width'], plotting_params['common_fig_height'] + 2))
            
        plt.subplots_adjust(hspace=0.25, wspace=0.2, top=0.89)
    
    if remove_only == True:
        ops = 'Remove'
    else:
        ops = 'Remove, Add, Flip'
    fig.suptitle(f"Nodes: {N}, Contexts: {nb_regimes}, Changed Links: {nb_changed_links}, Density: {density},\nCycles: {cycles_only}, Op.: {ops}", fontsize=plotting_params['title.size'])
    
    # Plot Union Metrics
    generate_time_conv_plot_union(
        all_configurations, sample_sizes=sample_sizes, regime_children_known=regime_children_known, 
        metrics_folder=metrics_folder, metrics_folder_fci=metrics_folder_fci, combined_plot=True, axs=axs[0], metrics_list=metrics_list, plot_avg=plot_avg, plot_fci=plot_fci,
        show_xticks=True
    )
    
    # Plot Regime Metrics
    generate_time_conv_plot_regimes(
        all_configurations, sample_sizes=sample_sizes, regime_children_known=regime_children_known, 
        metrics_folder=metrics_folder, combined_plot=True, axs=axs[1:].flatten(), metrics_list=metrics_list, plot_avg=plot_avg,
        show_xticks=True
    )
    if len(metrics_list) > 1:
        ax_for_handles = axs[0,0]
    else:
        ax_for_handles = axs[0]
    handles, labels = ax_for_handles.get_legend_handles_labels()

    fig_legend = plt.figure(figsize=(4, 3))
    fig_legend.legend(handles, labels, loc='lower center', ncol=4,  bbox_to_anchor=(0.5, -0.05), fontsize=plotting_params['legend.fontsize'])
    plt.tight_layout()
    # Save the legend as a separate file
    fig_legend.savefig(os.path.join(figure_path, 'legend_h' + '_'.join(str(known) for known in regime_children_known) + '.png'), bbox_inches='tight')
    plt.close(fig_legend)
    
    plt.savefig(os.path.join(figure_path, file_name + '_combined_' + '_'.join(metric for metric in metrics_list) + '_' + '_'.join(str(known) for known in regime_children_known) + '.png'), bbox_inches='tight')
    plt.close(fig)

def convert_to_string_list(s):
    # Check if the input is None or empty and return an empty list
    if not s:
        return []
    # Split the string on commas and strip any surrounding whitespace
    return [item.strip() for item in s.split(',')]

def convert_to_children_known(s):
    list = convert_to_string_list(s)

    for i in range(len(list)):
        if list[i] == 'False':
            list[i] = False
        if list[i] == 'True':
            list[i] = True
            
    return list

def convert_to_int_list(input_string):
    # Initialize an empty list to store the integers
    result = []
    
    # Split the input string by commas first, then iterate through each segment
    for item in input_string.split(','):
        # Further split by spaces to catch cases where spaces are used
        for subitem in item.split():
            # Attempt to convert each subitem to an integer
            try:
                # Convert subitem to integer and add to the result list
                result.append(int(subitem))
            except ValueError:
                # If a ValueError occurs (non-integer string), skip the subitem
                continue
    
    return result

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate configurations from a YAML file.")
    parser.add_argument('--yaml_path', type=str, help='Path to the YAML configuration file.')
    parser.add_argument('--metrics_list', type=convert_to_string_list, help='List of metrics to plot.')
    parser.add_argument('--sample_sizes', type=convert_to_int_list, required=False, help='Sample sizes to plot.')
    parser.add_argument('--metrics_folder', type=str, help='Folder where the metrics can be found.')
    parser.add_argument('--metrics_folder_fci', type=str, help='Folder where the metrics for FCI and CD-NOD can be found.')
    parser.add_argument('--plot_avg', type=str, required=False, help='Whether to plot averages.')
    parser.add_argument('--plot_known', type=convert_to_children_known, help='Which link assumptions to plot.')
    parser.add_argument('--plot_fci', type=str, help='Whether to plot fci.')
    

    args = parser.parse_args()
    config_path = args.yaml_path
    results_folder, all_configurations = generate_configurations(config_path)

    if args.plot_avg is None:
        args.plot_avg = 'True'

    if args.plot_fci is None:
        args.plot_fci = 'False'
    
    config_parameters = load_configurations(config_path)
    for node in config_parameters['N_values']:
        for factor in config_parameters['imbalance_factor']:
            for nb_links in config_parameters['nb_changed_links']:
                configs_for_nodes = [config for config in all_configurations if config[0][0] == node and config[0][12] == factor and config[0][6] == nb_links]

                if args.sample_sizes is not None:
                    sample_sizes = args.sample_sizes
                else:
                    sample_sizes = config_parameters['sample_sizes']

                print('sample size', sample_sizes)
                if args.plot_fci == 'True':
                    generate_fci_plots(
                    configs_for_nodes, 
                    sample_sizes=sample_sizes, 
                    regime_children_known=args.plot_known, 
                    metrics_folder=args.metrics_folder,
                    metrics_folder_fci=args.metrics_folder_fci,
                    metrics_list=args.metrics_list,
                    plot_avg=args.plot_avg,
                    plot_fci=args.plot_fci
                )
                else:
                    generate_combined_plots(
                        configs_for_nodes, 
                        sample_sizes=sample_sizes, 
                        regime_children_known=args.plot_known, 
                        metrics_folder=args.metrics_folder,
                        metrics_folder_fci=args.metrics_folder_fci,
                        metrics_list=args.metrics_list,
                        plot_avg=args.plot_avg,
                        plot_fci=args.plot_fci
                    )
