# -*- coding: utf-8 -*-
"""
@author: Rong Qian
@date: 2025/11/21
电子密度计算主函数
"""

from __future__ import print_function, division
# import sys
import iotbx.pdb
import iotbx.map_tools
import mmtbx.model
from mmtbx.maps import utils
import numpy as np
import re
import h5py

# import torch
# from io import StringIO
# 为 Python 2/3 兼容的导入：
try:
    from cStringIO import StringIO  # Python 2
except ImportError:
    from io import StringIO  # Python 3


def write_fake_p1_pdb(input_pdb, output_pdb, uc):
    """
    完全纯 Python 注入 fake P1:
    - 删除原 CRYST1 & SCALE
    - 插入新的 CRYST1, SCALE1-3
    """
    a, b, c, alpha, beta, gamma, sg = uc

    # 读取旧的 PDB 内容
    with open(input_pdb, "r") as f:
        lines = f.readlines()

    # 去掉已有 CRYST1 / SCALE
    new_lines = [
        ln for ln in lines
        if not ln.startswith("CRYST1") and not ln.startswith("SCALE")
    ]

    # 构造新的 CRYST1 + SCALE
    cryst1 = "CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f} {}\n".format(
        a, b, c, alpha, beta, gamma, sg
    )
    scale1 = "SCALE1    {:8.6f}  0.000000  0.000000        0.00000\n".format(1.0/a)
    scale2 = "SCALE2    0.000000  {:8.6f}  0.000000        0.00000\n".format(1.0/b)
    scale3 = "SCALE3    0.000000  0.000000  {:8.6f}        0.00000\n".format(1.0/c)


    # 写回：将 SCALE 和 CRYST1 插入文件最前面（与你 sed 行为一致）
    with open(output_pdb, "w") as f:
        f.write(cryst1)
        f.write(scale1)
        f.write(scale2)
        f.write(scale3)
        f.writelines(new_lines)

def build_fake_p1_pdb_in_memory(inpdbfile, uc):
    """
    内存版 fake P1：
    - 删除原 CRYST1 & SCALE
    - 插入新的 CRYST1, SCALE1-3
    - 返回 StringIO，不落盘
    """
    a, b, c, alpha, beta, gamma, sg = uc

    fake_lines = []
    fake_lines.append("CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f %s\n" %
                      (a, b, c, alpha, beta, gamma, sg))
    fake_lines.append("SCALE1    %8.6f  0.000000  0.000000        0.00000\n" % (1.0/a))
    fake_lines.append("SCALE2    0.000000  %8.6f  0.000000        0.00000\n" % (1.0/b))
    fake_lines.append("SCALE3    0.000000  0.000000  %8.6f        0.00000\n" % (1.0/c))

    try:
        with open(inpdbfile, "r") as f:
            original_lines = f.readlines()
    except Exception as e:
        raise RuntimeError("无法读取PDB文件 %s: %s" % (inpdbfile, e))

    # 过滤掉原有的 CRYST1 和 SCALE 行
    filtered_lines = [
        ln for ln in original_lines
        if not ln.startswith("CRYST1") and not ln.startswith("SCALE")
    ]

    # 合并所有行
    all_lines = fake_lines + filtered_lines
    new_text = "".join(all_lines)
    
    # 确保返回的是 StringIO 对象
    return StringIO(new_text)

def convert_to_pdb_str(xyz, atom_name, elem, atom_seq_num, res_name, res_seq_num,
                       chain_id='A', head='HETATM', alter_loc_indicator=' ',
                       insert_of_res=' ', occupancy=1.0, temperature_factor=1.0, seg_id=''):
    """
    Convert input info to a string in PDB format.
    """
    assert len(res_name) <= 3, 'The length of residue name must be less than 3.'
    assert len(atom_name) <= 4, 'The length of atom name must be less than 4.'
    assert len(chain_id) == 1, 'Chain identifier can only be a uppercase letter.'
    assert len(alter_loc_indicator) == 1, 'Alternate location indicator can only be a uppercase letter'
    assert len(insert_of_res) == 1, 'Code for insertions of residues can only be a uppercase letter'
    assert head in ['ATOM', 'HETATM']

    x, y, z = [('%.3f' % round(i, 3)).rjust(8) for i in xyz]

    # 'ATOM' or 'HETATM'
    head = head.ljust(6)
    atom_seq_num = str(atom_seq_num).rjust(5)
    res_name = res_name.rjust(3)
    atom_name = atom_name.ljust(4)
    res_seq_num = str(res_seq_num).rjust(4)
    occupancy = ('%.2f' % occupancy).rjust(6)
    temperature_factor = ('%.2f' % temperature_factor).rjust(6)
    seg_id = seg_id.ljust(4)
    elem = elem.rjust(2)
    
    # Python 2.7 compatible string formatting
    pdb_str = '{}{} {}{}{} {}{}{}   {}{}{}{}{}      {}{}  \n'.format(
        head, atom_seq_num, atom_name, alter_loc_indicator, res_name, 
        chain_id, res_seq_num, insert_of_res, x, y, z, occupancy, 
        temperature_factor, seg_id, elem)
    return pdb_str

def convert_xplor_to_pdb(xplor_file, output_pdb=None, lower_limit=1e-4, upper_limit=float('inf')):
    """
    将XPLOR文件转换为PDB格式（替代外部脚本调用）
    """
    if output_pdb is None:
        output_pdb = xplor_file.replace('.xplor', '.pdb')
    
    all_pdb_str = ''
    xr = XplorReader(xplor_file)
    
    for idx_a, a in enumerate(xr.density_array):
        for idx_b, b in enumerate(a):
            for idx_c, c in enumerate(b):
                if lower_limit <= c <= upper_limit:
                    xyz = xr.idx2pos([idx_a, idx_b, idx_c])[0]
                    pdb_str = convert_to_pdb_str(xyz, 'DU', 'DU', 1, 'DUM', 1,
                                    chain_id='A', head='HETATM', alter_loc_indicator=' ',
                                    insert_of_res=' ', occupancy=1.0, temperature_factor=float(c), seg_id='')
                    all_pdb_str += pdb_str
    
    if not all_pdb_str:
        raise RuntimeError('No value output!')

    with open(output_pdb, 'w') as f:
        f.write(all_pdb_str)
    print("Converted {} to {}".format(xplor_file, output_pdb))
    return output_pdb

class XplorReader:
    """
    Xplor data parser.
    """

    def __init__(self, xplor):
        self.xplor = xplor
        self.lines = self._read_lines(xplor)
        self.map_info_dict, self.array_start_idx = self._read_map_info(self.lines)
        self.move_vector = self._get_move_vector()
        self.scale_vector = self._get_scale_vector()
        self.transfer_matrix = self._get_transfer_matrix()
        self.density_array = self._get_density_array()

    @staticmethod
    def _read_lines(xplor):
        with open(xplor) as f:
            lines = f.readlines()
        lines = [i for i in lines if i.strip()]
        return lines

    @staticmethod
    def _read_map_info(lines):
        map_info_dict = {}
        for idx, line in enumerate(lines):
            if len(re.findall(r'\d+', line)) == 9:
                int9_values = [int(i) for i in line.split()]
                int9_names = ['NA', 'AMIN', 'AMAX', 'NB', 'BMIN', 'BMAX', 'NC', 'CMIN', 'CMAX']
                for int9_name, int9_value in zip(int9_names, int9_values):
                    map_info_dict[int9_name] = int9_value

                float6_values = [float(i) for i in lines[idx + 1].split()]
                float6_names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
                for float6_name, float6_value in zip(float6_names, float6_values):
                    map_info_dict[float6_name] = float6_value
                array_start_idx = idx + 3
                break
        else:
            raise Exception('Read xplor failed: can not read map info.')
        return map_info_dict, array_start_idx

    def _get_density_array(self):
        cba_array = []
        ba_array = []
        for line in self.lines[self.array_start_idx:]:
            if line.startswith('  '):
                if ba_array:
                    cba_array.append(ba_array)
                if line.strip() == '-9999':
                    break
                ba_array = []
            else:
                num = int(len(line) / 12)
                for i in range(num):
                    density_value = float(line.rstrip()[i * 12:(i + 1) * 12])
                    ba_array.append(density_value)
        cba_array = np.array(cba_array)
        len_a = self.map_info_dict['AMAX'] - self.map_info_dict['AMIN'] + 1
        len_b = self.map_info_dict['BMAX'] - self.map_info_dict['BMIN'] + 1
        len_c = self.map_info_dict['CMAX'] - self.map_info_dict['CMIN'] + 1
        cba_array = cba_array.reshape((len_c, len_b, len_a))
        abc_array = cba_array.transpose(2, 1, 0)
        return abc_array

    def _get_move_vector(self):
        move_vector = [self.map_info_dict['AMIN'],
                       self.map_info_dict['BMIN'],
                       self.map_info_dict['CMIN']]
        move_vector = np.array(move_vector)
        return move_vector

    def _get_scale_vector(self):
        scale_vector = [self.map_info_dict['NA'],
                        self.map_info_dict['NB'],
                        self.map_info_dict['NC']]
        scale_vector = np.array(scale_vector)
        return scale_vector

    def _get_frac_to_real_matrix(self):
        a, b, c, alpha, beta, gamma = [self.map_info_dict[i]
                                       for i in ('a', 'b', 'c', 'alpha', 'beta', 'gamma')]
        alpha, beta, gamma = [i / 180 * np.pi for i in (alpha, beta, gamma)]

        m1 = [a, b * np.cos(gamma), c * np.cos(beta)]
        m2 = [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)]
        m33 = c * np.sqrt(
            1 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma) - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(
                gamma) ** 2) / np.sin(gamma)
        m3 = [0, 0, m33]
        M_f2r = np.round(np.array([m1, m2, m3]), 6)
        return M_f2r

    def _get_real_to_frac_matrix(self):
        M_f2r = self._get_frac_to_real_matrix()
        M_r2f = np.round(np.linalg.inv(M_f2r), 6)
        return M_r2f

    def _get_transfer_matrix(self):
        M_r2f = self._get_real_to_frac_matrix()
        transfer_matrix = M_r2f.T * self.scale_vector
        return transfer_matrix

    def idx2pos(self, index_array):
        index_array = np.array(index_array)

        if len(index_array.shape) == 1:
            index_array = np.expand_dims(index_array, 0)

        position_array = np.dot((index_array + self.move_vector),
                                np.linalg.inv(self.transfer_matrix))
        return position_array

def compute_density_physics(rho, spacing):
    """
    计算电子密度的物理量（梯度、一阶模、Laplacian、Hessian eigenvalues）
    
    输入：
        rho:      3D numpy array, shape = (A, B, C)
        spacing:  tuple/list of voxel spacing (sx, sy, sz) in Å
                  一般来自 (a/NA, b/NB, c/NC)

    输出：
        grad:      (A, B, C, 3)   梯度向量 (∂ρ/∂x, ∂ρ/∂y, ∂ρ/∂z)
        grad_norm: (A, B, C)      梯度模长
        lap:       (A, B, C)      Laplacian  ∂²ρ/∂x² + ∂²ρ/∂y² + ∂²ρ/∂z²
        eig:       (A, B, C, 3)   Hessian 的 3 个特征值（排序好的）
    """

    sx, sy, sz = spacing  # 真实 voxel 间距（Å）

    # -------------------------------
    # 1) 一阶导数（gradient）
    # -------------------------------
    gx, gy, gz = np.gradient(rho, sx, sy, sz)

    grad = np.stack([gx, gy, gz], axis=-1)
    grad_norm = np.sqrt(gx * gx + gy * gy + gz * gz)

    # -------------------------------
    # 2) 二阶导（对 gx, gy, gz 再求 gradient）
    # -------------------------------
    gxx, gxy, gxz = np.gradient(gx, sx, sy, sz)
    _,   gyy, gyz = np.gradient(gy, sx, sy, sz)
    _,    _, gzz  = np.gradient(gz, sx, sy, sz)

    # Laplacian = ∂²ρ/∂x² + ∂²ρ/∂y² + ∂²ρ/∂z²
    lap = gxx + gyy + gzz

    # -------------------------------
    # 3) Hessian + 特征值
    # -------------------------------
    # A, B, C = rho.shape
    # eig = np.zeros((A, B, C, 3))

    # for i in range(A):
    #     for j in range(B):
    #         for k in range(C):

    #             H = np.array([
    #                 [gxx[i,j,k], gxy[i,j,k], gxz[i,j,k]],
    #                 [gxy[i,j,k], gyy[i,j,k], gyz[i,j,k]],
    #                 [gxz[i,j,k], gyz[i,j,k], gzz[i,j,k]],
    #             ])

    #             eig[i,j,k] = np.linalg.eigvalsh(H)

    # 优化版：矢量化计算
    H = np.stack([
            np.stack([gxx, gxy, gxz], axis=-1),
            np.stack([gxy, gyy, gyz], axis=-1),
            np.stack([gxz, gyz, gzz], axis=-1)
        ],
        axis=-2
    )  # shape = (*,3,3)

    # 对所有点一次性求 eigvalsh
    eig = np.linalg.eigvalsh(H)


    return grad, grad_norm, lap, eig

def convert_xplor_to_npz_single_res(xplor_file, output_npz=None, 
                         density_threshold=3.0, resolution=None):
    xr = XplorReader(xplor_file)
    rho = xr.density_array

     # ========= 计算 spacing (sx, sy, sz) =========
    a = xr.map_info_dict['a']
    b = xr.map_info_dict['b']
    c = xr.map_info_dict['c']

    NA, NB, NC = xr.scale_vector  # 分别是 x/y/z 方向的格点数

    spacing = (a / NA, b / NB, c / NC)


    # 计算物理量
    grad, grad_norm, lap, eig = compute_density_physics(rho, spacing)

    xyz_list = []
    rho_list = []
    grad_list = []
    gradnorm_list = []
    lap_list = []
    eig_list = []

    A, B, C = rho.shape
    for ia in range(A):
        for ib in range(B):
            for ic in range(C):
                v = rho[ia, ib, ic]
                if v >= density_threshold:
                    xyz = xr.idx2pos([ia, ib, ic])[0]
                    xyz_list.append(xyz)
                    rho_list.append(v)
                    grad_list.append([
                        grad[ia, ib, ic, 0],
                        grad[ia, ib, ic, 1],
                        grad[ia, ib, ic, 2]
                    ])
                    gradnorm_list.append(grad_norm[ia,ib,ic])
                    lap_list.append(lap[ia,ib,ic])
                    eig_list.append(eig[ia,ib,ic])

    # 转为 float32（非常重要）
    return dict(
        xyz=np.array(xyz_list, dtype=np.float32),
        rho=np.array(rho_list, dtype=np.float16),
        grad=np.array(grad_list, dtype=np.float16),
        grad_norm=np.array(gradnorm_list, dtype=np.float16),
        lap=np.array(lap_list, dtype=np.float16),
        eig=np.array(eig_list, dtype=np.float16),
        resolution=np.float16(resolution)
    )


def save_multires_hdf5(output_file, all_res_data):
    """
    保存多分辨率 ED 点云到 HDF5
    all_res_data 是这样的结构：
    {
        "2.7": { "xyz":..., "rho":..., "grad":..., "lap":..., "eig":... },
        "3.5": { ... }
    }
    """

    with h5py.File(output_file, "w") as f:
        # 保存分辨率列表
        resolutions = np.array([float(r) for r in all_res_data.keys()], dtype=np.float16)
        f.create_dataset("resolutions", data=resolutions)

        # 顶级 group：data
        grp = f.create_group("data")

        # 遍历每个分辨率
        for res, res_data in all_res_data.items():
            res_grp = grp.create_group(str(res))   # e.g. "2.7"

            # 保存每个物理量，开启压缩
            res_grp.create_dataset("xyz", data=res_data["xyz"], 
                                   compression="gzip", compression_opts=2, shuffle=True)

            res_grp.create_dataset("rho", data=res_data["rho"], 
                                   compression="gzip", compression_opts=2, shuffle=True)

            res_grp.create_dataset("grad", data=res_data["grad"], 
                                   compression="gzip", compression_opts=2, shuffle=True)
            
            res_grp.create_dataset("grad_norm", data=res_data["grad_norm"], 
                                   compression="gzip", compression_opts=2, shuffle=True)

            res_grp.create_dataset("lap", data=res_data["lap"], 
                                   compression="gzip", compression_opts=2, shuffle=True)

            res_grp.create_dataset("eig", data=res_data["eig"], 
                                   compression="gzip", compression_opts=2, shuffle=True)
            
    print("HDF5 saved:", output_file)


def generate_xplor_map_from_pdb(
    inpdbfile,
    resolution_needed=[2.0, 2.5, 3.0, 3.5, 4.0],
    grid_value=0.5,
    mtz_res=1.0,
    need_fake_p1=True,
    uc=(50, 50, 50, 90, 90, 90, "P 1"),
    need_0_b=True,
    set_b_base_value="no_need",
    save_pdb=False,
    density_threshold=3.0,
    keep_xplor=False,
    return_aux=False,
    npz=False
):
    prefix = inpdbfile[:inpdbfile.rfind(".pdb")]

    generated_maps = []
    output_pdbs = []
    all_res_data = {}

    # =========== fake p1 ==============
    if need_fake_p1:
        pdb_buf = build_fake_p1_pdb_in_memory(inpdbfile, uc)
        pdb_lines = pdb_buf.getvalue().splitlines(True)
        pdb_inp = iotbx.pdb.input(source_info=None, lines=pdb_lines)
    else:
        pdb_inp = iotbx.pdb.input(file_name=inpdbfile)

    model = mmtbx.model.manager(model_input=pdb_inp)

    if need_0_b:
        model.set_b_iso(model.get_b_iso() * 0)

    xrs = model.get_xray_structure()
    sites_cart = xrs.sites_cart()

    if set_b_base_value.isdigit():
        f_calc = xrs.structure_factors(
            d_min=mtz_res, b_base=float(set_b_base_value)
        ).f_calc()
    else:
        f_calc = xrs.structure_factors(d_min=mtz_res).f_calc()

    # ============ 多分辨率循环 ============
    for i_res in resolution_needed:
        f_calc_res = f_calc.resolution_filter(d_min=i_res)
        fft_map = f_calc_res.fft_map(
            resolution_factor=round(grid_value / i_res, 2),
            d_min=i_res,
        )
        fft_map.apply_sigma_scaling()
        map_data = fft_map.real_map_unpadded()

        # xplor_name = f"{prefix}_{i_res}_map.xplor"
        xplor_name = "{}_{}_map.xplor".format(prefix, i_res)


        utils.write_xplor_map(
            sites_cart=sites_cart,
            unit_cell=f_calc_res.unit_cell(),
            map_data=map_data.as_double(),
            n_real=fft_map.n_real(),
            file_name=xplor_name,
            buffer=8.2,
        )

        if keep_xplor:
            generated_maps.append(xplor_name)

        # ========== PDB 输出（可选）==========
        if save_pdb:
            pdb_name = convert_xplor_to_pdb(
                xplor_name, lower_limit=density_threshold
            )
            output_pdbs.append(pdb_name)

        # ========= 计算单分辨率点云数据（不落盘）=========
        res_data = convert_xplor_to_npz_single_res(
            xplor_name,
            density_threshold=density_threshold,
            resolution=i_res,
        )

        all_res_data[str(i_res)] = res_data

        # ========= 删除 xplor（如果不留）=========
        if not keep_xplor:
            import os
            os.remove(xplor_name)

    # =========== 写多分辨率 HDF5 ==============
    h5_file = prefix + "_multires_pointcloud.h5"
    save_multires_hdf5(h5_file, all_res_data)
    print("Saved HDF5:", h5_file)

    # =========== 可选：保存 npz（调试用）==========
    if npz:
        final_npz = prefix + "_multires_pointcloud.npz"
        np.savez_compressed(
            final_npz,
            resolutions=np.array([float(r) for r in all_res_data.keys()], dtype=np.float16),
            data=all_res_data,
        )
        print("Saved NPZ:", final_npz)

    # =========== 返回值逻辑 ==============
    if return_aux:
        return h5_file, final_npz, generated_maps, output_pdbs

    return h5_file


if __name__ == "__main__":
    try:
        # h5, npz, xplor, pdbs = generate_xplor_map_from_pdb(
        #     "5n10.pdb",
        #     save_pdb=True,
        #     keep_xplor=True,
        #     return_aux=True
        # )

        h5 = generate_xplor_map_from_pdb(
            "5n10.pdb",
            save_pdb=False,
            keep_xplor=False,
            return_aux=False
        )

        print("\n测试完成!")
        print("HDF5 文件:", h5)

    except Exception as e:
        print("处理出错:", e)

