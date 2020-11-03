import numpy as np
from sklearn.decomposition import PCA


class model:

    def PCA(self, data, n_components):

        if data.ndim > 2:
            data = data.reshape(-1, data.shape[2])
        pca = PCA(n_components)
        data = pca.fit_transform(data)
        return data

    def PLS_remove_background(self, hsi, n_components, points, show_hsi_band=10):
        from sklearn.cross_decomposition import PLSRegression
        from matplotlib import pyplot as plt
        pls = PLSRegression(n_components)
        plt.figure(f"點選目標點共:{points}個點,分{n_components}類")
        plt.imshow(hsi[:, :, show_hsi_band])
        pos = np.array(plt.ginput(points), 'i')
        d = [hsi[y, x] for x, y in pos]
        d = np.array(d)
        plt.close()
        y = []
        classify = 0
        for i in range(points):
            check_value = n_components
            if (i / n_components) == 0:
                classify = classify + 1
            y[i] = classify
        y = np.array(y)
        one_hot = np.eye(n_components + 1)[y]
        d = this.PCA(d, n_components)
        hsi = this.PCA(hsi, n_components)
        pls_model = pls.fit(d, one_hot)
        remove_background_res = pls_model.predict(
            hsi.reshape(-1, n_components))
        remove_background_res = np.argmax(
            remove_background_res, -1).reshape(hsi.shape[0], hsi.shape[1])
        plt.figure("background_remove_result")
        plt.imshow(remove_background_res)
        return remove_background_res
