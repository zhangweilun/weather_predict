from visdom import Visdom


if __name__ == '__main__':
    vis = Visdom()

    vis.line(X=np.array(x), Y=np.column_stack((np.array(y), np.array(z))), opts=dict(showlegend=True))