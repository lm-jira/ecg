import click
import importlib


@click.command()
@click.option('--opt',
              type=click.Choice(['train', 'test', 'export', 'predict']),
              default='train',
              help='train or test the model')
@click.option('--predict_img', type=click.File(),
              help='directory of input csv files')
@click.option('--nb_epochs', default=100)
@click.option('--weight', default=0.1)
@click.option('--logname', default="")
@click.option('--gpu', default=2)
@click.option('--model', default="effanogan")
def run(opt, predict_img, nb_epochs, weight, logname, gpu, model):

    mod_name = "executor.{}_{}".format(model, opt)
    mod = importlib.import_module(mod_name)

    if opt == "train":
        mod.run(nb_epochs, weight, logname, gpu)
    elif opt == "test":
        mod.run(logname, gpu)
    elif opt == "export":
        #        mod.run()
        pass
    elif opt == 'predict':
        #        mod.run(predict_img)
        pass


if __name__ == "__main__":
    run()
