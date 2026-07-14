def build_dataset(scene_ls, runs, zmap, runs_ls, scene, bs):
    dataset = {}

    for i_scene in scene_ls:
        dataset[i_scene] = {}

        for i_run in range(runs):
            dataset[i_scene][i_run] = {}

            cur_train_x, cur_test_x = [], []
            cur_train_y, cur_test_y = [], []
            cur_train_scene, cur_test_scene = [], []

            for i_idx in range(len(zmap)):
                if runs_ls[i_idx] != i_run and scene[i_idx] != i_scene:
                    cur_train_x.append(zmap[i_idx])
                    cur_train_y.append(bs[i_idx])
                    cur_train_scene.append(f"{scene[i_idx]}-{runs_ls[i_idx]}")

                elif runs_ls[i_idx] == i_run and scene[i_idx] == i_scene:
                    cur_test_x.append(zmap[i_idx])
                    cur_test_y.append(bs[i_idx])
                    cur_test_scene.append(f"{scene[i_idx]}-{runs_ls[i_idx]}")

            dataset[i_scene][i_run]["train_x"] = cur_train_x
            dataset[i_scene][i_run]["train_y"] = cur_train_y
            dataset[i_scene][i_run]["test_x"] = cur_test_x
            dataset[i_scene][i_run]["test_y"] = cur_test_y
            dataset[i_scene][i_run]["train_scene"] = cur_train_scene
            dataset[i_scene][i_run]["test_scene"] = cur_test_scene

    return dataset
