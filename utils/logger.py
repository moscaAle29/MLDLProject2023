import pytorch_lightning
import wandb

class Logger(pytorch_lightning.loggers.WandbLogger):
    def __init__(self, project, name, log_model = False, save_dir = None):
        super(Logger, self).__init__(name = name, project = project, log_model = log_model, save_dir = save_dir)
    
    def save(self, obj):
        return wandb.save(obj)

    @staticmethod
    def restore(name, run_path, root):
        return wandb.restore(name=name, run_path=run_path, root=root)



def get_job_name(args):
    job_name = ""
    if args.setting == 'federated':
        if args.algorithm == 'SiloBN':
            job_name += "SBN_"
        job_name += f"{args.setting}_cl{args.clients_per_round}_e{args.num_epochs}_"

    if args.setting == 'centralized':
        job_name += f"{args.setting}_cl{1}_e{args.num_epochs}_"

    if args.flip:
        job_name += "flip_"
    if args.rrc_transform:
        job_name += "rrc_"
    if args.random_rotation:
        job_name += "random_rotation_"
    #if args.dom_gen is not None:
    #    job_name += f"{args.dom_gen}_"
    #if args.dd_batch_size:
    #    job_name += f"ddbs{args.dd_batch_size}_"
    if args.cv2_transform:
        job_name += "cv2_"
    if args.jitter:
        job_name += "jitter_"
    if args.use_test_resize:
        job_name += "testResize_"
    if args.domain_adapt is not None:
        job_name += f"{args.domain_adapt}_"


    #job_name += f"lr{args.lr}_rs{args.random_seed}_{args.clients_type}"
    #if args.framework == 'federated':
    #    job_name += f"_SerOpt:{args.server_opt}_lr{args.server_lr}_m{args.server_momentum}"

    #if args.custom_lr_param:
    #    job_name += "_customlrparam"
    #if args.dataset == 'idda':
    #    job_name += f"_{args.setting_type}"
    #job_name += f"_{args.name}"

    return job_name


def get_group_name(args):
    group_name = args.algorithm
    if args.dom_gen is not None:
        group_name += f"_{args.dom_gen}"
    return group_name


def get_project_name(args):
    return f"{args.setting}_{args.dataset}_{args.dataset2}"