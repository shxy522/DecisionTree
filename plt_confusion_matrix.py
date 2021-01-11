import matplotlib.pyplot as plt

def plt_confusion_matrix(confusion,labels,save_path):
    confusion = confusion
    # 热度图，后面是指定的颜色块，可设置其他的不同颜色
    plt.imshow(confusion, cmap=plt.cm.Blues)
    # ticks 坐标轴的坐标点
    # label 坐标轴标签说明
    indices = range(len(confusion))
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    plt.xticks(indices, labels)
    plt.yticks(indices, labels)

    plt.colorbar()

    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('混淆矩阵')

    # plt.rcParams两行是用于解决标签不能显示汉字的问题
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 显示数据
    for first_index in range(len(confusion)):    #第几行
        for second_index in range(len(confusion[first_index])):    #第几列
            plt.text(first_index, second_index, int(confusion[second_index][first_index]))
    # 在matlab里面可以对矩阵直接imagesc(confusion)
    # 显示
    plt.savefig(save_path)
    plt.show()