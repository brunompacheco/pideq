import click
import torch
from pideq.deq.solvers import anderson, broyden, forward_iteration

from pideq.net import PINN, PIDEQ
from pideq.trainer import PINNTrainer, PIDEQTrainer


def experiment_1():
    print('=== EXPERIMENT 1 ===')
    PINNTrainer(
        PINN(2., n_nodes=20),
        epochs=5e4,
        wandb_group=f'PINN-baseline',
    ).run()

def experiment_2():
    print('=== EXPERIMENT 2 ===')
    PIDEQTrainer(
        PIDEQ(2., n_states=80),
        epochs=5e4,
        wandb_group=f'PIDEQ-baseline',
    ).run()

def experiment_3(ns_states=[40, 20, 10, 5, 2,]):
    print('=== EXPERIMENT 3 ===')
    for n_states in ns_states:
        PIDEQTrainer(
            PIDEQ(2., n_states=n_states),
            epochs=5e4,
            wandb_group=f'PIDEQ-#z={n_states}',
        ).run()

def experiment_4(ns_hidden=[1, 2,]):
    print('=== EXPERIMENT 4 ===')
    for n_hidden in ns_hidden:
        PIDEQTrainer(
            PIDEQ(2., n_states=5, n_hidden=n_hidden),
            epochs=5e4,
            wandb_group=f'PIDEQ-#hidden={n_hidden}',
        ).run()

def experiment_5(jac_lambdas=[0.1, 2]):
    print('=== EXPERIMENT 5 ===')
    for jac_lambda in jac_lambdas:
        PIDEQTrainer(
            PIDEQ(2., n_states=5),
            epochs=5e4,
            jac_lamb=jac_lambda,
            wandb_group=f'PIDEQ-#jac_lamb={jac_lambda}',
        ).run()

def experiment_6(solvers=[forward_iteration, broyden]):
    print('=== EXPERIMENT 6 ===')
    for solver in solvers:
        PIDEQTrainer(
            PIDEQ(2., n_states=5, solver=solver),
            epochs=5e4,
            wandb_group=f'PIDEQ-#solver={solver.__name__}',
        ).run()

def experiment_7(epss=[1e-2, 1e-6]):
    print('=== EXPERIMENT 7 ===')
    for eps in epss:
        PIDEQTrainer(
            PIDEQ(2., n_states=5, solver=forward_iteration, solver_kwargs={'threshold': 200, 'eps': eps}),
            epochs=5e4,
            wandb_group=f'PIDEQ-#eps={eps:.0e}',
        ).run()

def experiment_8():
    print('=== EXPERIMENT 8 ===')
    PIDEQTrainer(
        PIDEQ(2., n_states=5, solver=broyden),
        epochs=5e4,
        lr_scheduler='MultiStepLR',
        lr_scheduler_params={'milestones': [30000, 40000]},
        wandb_group=f'PIDEQ-#step_decay',
    ).run()

def experiment_9():
    print('=== EXPERIMENT 9 ===')
    PINNTrainer(
        PINN(2., n_hidden=2, n_nodes=5),
        epochs=5e4,
        wandb_group=f'PINN-baseline-small',
    ).run()

@click.command()
@click.option('-n', '--n-runs', default=1, show_default=True, type=click.INT,
              help=("Number of times each experiment is run, i.e., number of "
                    "networks trained with each experiment's configuration."))
@click.argument('experiment', nargs=-1)
def main(n_runs, experiment):
    """Runs `EXPERIMENT` for `--n-runs` time(s).

    `EXPERIMENT` can be either a single number, an interval using dash notation
    (`2-5` means that experiments 2, 3, 4 and 5 will be executed), or any
    sequence of numbers and intervals, like `1 3-5 8`.
    """
    for exp in experiment:
        try:
            exps = [int(exp),]
        except ValueError:
            # exp must be an interval, then
            start, stop = exp.split('-')
            exps = range(int(start), int(stop)+1)

        for exp_n in exps:
            for _ in range(n_runs):
                experiment = eval(f"experiment_{exp_n}")
                experiment()


if __name__ == '__main__':
    main()
