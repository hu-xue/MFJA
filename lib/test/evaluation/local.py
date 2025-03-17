from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = 'E:/dataset/got10k_lmdb'
    settings.got10k_path = 'E:/dataset/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = 'E:/dataset/itb'
    settings.lasot_extension_subset_path_path = 'E:/dataset/lasot_extension_subset'
    settings.lasot_lmdb_path = 'E:/dataset/lasot_lmdb'
    settings.lasot_path = 'E:/dataset/lasot'
    settings.network_path = 'E:/project/trust_fusion/MFJA/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = 'E:/dataset/nfs'
    settings.otb_path = 'E:/dataset/otb'
    settings.prj_dir = 'E:/project/trust_fusion/MFJA'
    settings.result_plot_path = 'E:/project/trust_fusion/MFJA/output/test/result_plots'
    settings.results_path = 'E:/project/trust_fusion/MFJA/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = 'E:/project/trust_fusion/MFJA/output'
    settings.segmentation_path = 'E:/project/trust_fusion/MFJA/output/test/segmentation_results'
    settings.tc128_path = 'E:/dataset/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = 'E:/dataset/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = 'E:/dataset/trackingnet'
    settings.uav_path = 'E:/dataset/uav'
    settings.vot18_path = 'E:/dataset/vot2018'
    settings.vot22_path = 'E:/dataset/vot2022'
    settings.vot_path = 'E:/dataset/VOT2019'
    settings.youtubevos_dir = ''

    return settings

