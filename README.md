# 目标检测-垃圾分类

基于open-cv的垃圾分类检测，detect.py中有三种目标检测方法（KNN、MOG2、GrabCut）各有特点，适用于不同场景。

选择建议：

​	•	**实时视频监控场景**：优先使用 KNN 或 MOG2。

​	•	KNN 更适合背景动态变化明显的场景。

​	•	MOG2 更适合背景稍复杂但稳定性较高的场景。

​	•	**静态图像或单帧视频处理**：使用 GrabCut，获得高质量的目标分割效果。

​	•	**动态摄像头中前景提取**：根据实际情况选择 KNN 或 MOG2，结合形态学操作提升精度。



添加了用户数据服务和GUI，用户可登录并上传图片，如果成功上传五次可回收物则奖励用户奖励动画。

建表参考sql：

```sql
/*
 Navicat Premium Data Transfer

 Source Server         : test
 Source Server Type    : MySQL
 Source Server Version : 80040 (8.0.40)
 Source Host           : localhost:3306
 Source Schema         : recycle

 Target Server Type    : MySQL
 Target Server Version : 80040 (8.0.40)
 File Encoding         : 65001

 Date: 13/12/2024 21:04:40
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for recyclable_logs
-- ----------------------------
DROP TABLE IF EXISTS `recyclable_logs`;
CREATE TABLE `recyclable_logs` (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(255) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `file_path` varchar(255) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `is_recyclable` tinyint(1) DEFAULT NULL,
  `upload_time` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=23 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- ----------------------------
-- Records of recyclable_logs
-- ----------------------------
BEGIN;
INSERT INTO `recyclable_logs` (`id`, `username`, `file_path`, `is_recyclable`, `upload_time`) VALUES (12, 'user1', '/Users/apple/Desktop/素材/垃圾分类/易拉罐.JPG', 1, '2024-12-13 20:33:17');
INSERT INTO `recyclable_logs` (`id`, `username`, `file_path`, `is_recyclable`, `upload_time`) VALUES (13, 'user1', '/Users/apple/Desktop/素材/垃圾分类/易拉罐.JPG', 1, '2024-12-13 20:34:33');
INSERT INTO `recyclable_logs` (`id`, `username`, `file_path`, `is_recyclable`, `upload_time`) VALUES (14, 'user1', '/Users/apple/Desktop/素材/垃圾分类/易拉罐.JPG', 1, '2024-12-13 20:34:36');
INSERT INTO `recyclable_logs` (`id`, `username`, `file_path`, `is_recyclable`, `upload_time`) VALUES (15, 'user1', '/Users/apple/Desktop/素材/垃圾分类/易拉罐.JPG', 1, '2024-12-13 20:34:39');
INSERT INTO `recyclable_logs` (`id`, `username`, `file_path`, `is_recyclable`, `upload_time`) VALUES (16, 'user1', '/Users/apple/Desktop/素材/垃圾分类/易拉罐.JPG', 1, '2024-12-13 20:34:40');
INSERT INTO `recyclable_logs` (`id`, `username`, `file_path`, `is_recyclable`, `upload_time`) VALUES (17, 'user1', '/Users/apple/Desktop/素材/垃圾分类/易拉罐.JPG', 1, '2024-12-13 20:34:42');
INSERT INTO `recyclable_logs` (`id`, `username`, `file_path`, `is_recyclable`, `upload_time`) VALUES (18, 'user1', '/Users/apple/Desktop/素材/垃圾分类/塑料袋.jpg', 1, '2024-12-13 20:35:36');
INSERT INTO `recyclable_logs` (`id`, `username`, `file_path`, `is_recyclable`, `upload_time`) VALUES (19, 'user1', '/Users/apple/Desktop/素材/垃圾分类/塑料袋.jpg', 1, '2024-12-13 20:35:38');
INSERT INTO `recyclable_logs` (`id`, `username`, `file_path`, `is_recyclable`, `upload_time`) VALUES (20, 'user1', '/Users/apple/Desktop/素材/垃圾分类/塑料袋.jpg', 1, '2024-12-13 20:35:39');
INSERT INTO `recyclable_logs` (`id`, `username`, `file_path`, `is_recyclable`, `upload_time`) VALUES (21, 'user1', '/Users/apple/Desktop/素材/垃圾分类/塑料袋.jpg', 1, '2024-12-13 20:35:41');
INSERT INTO `recyclable_logs` (`id`, `username`, `file_path`, `is_recyclable`, `upload_time`) VALUES (22, 'user1', '/Users/apple/Desktop/素材/垃圾分类/塑料袋.jpg', 1, '2024-12-13 20:35:43');
COMMIT;

-- ----------------------------
-- Table structure for users
-- ----------------------------
DROP TABLE IF EXISTS `users`;
CREATE TABLE `users` (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(255) COLLATE utf8mb4_general_ci NOT NULL,
  `password` varchar(255) COLLATE utf8mb4_general_ci NOT NULL,
  `recycle_count` int DEFAULT '0',
  `recyclable_count` int DEFAULT '0',
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- ----------------------------
-- Records of users
-- ----------------------------
BEGIN;
INSERT INTO `users` (`id`, `username`, `password`, `recycle_count`, `recyclable_count`) VALUES (1, 'test_user', 'password123', 0, 0);
INSERT INTO `users` (`id`, `username`, `password`, `recycle_count`, `recyclable_count`) VALUES (2, 'user1', 'pass1', 105, 10);
INSERT INTO `users` (`id`, `username`, `password`, `recycle_count`, `recyclable_count`) VALUES (3, 'user2', 'pass2', 7, 0);
COMMIT;

SET FOREIGN_KEY_CHECKS = 1;
```

### 运行前向传播测试

运行 `inference.py` 可测试先前传播。修改 `config.py` 中 `DefaultConfig.InferWithGPU` 参数即可切换前向传播使用 GPU 还是 CPU。对 `inference.py` 文件稍加修改即可对自己的图片进行分类。

## 训练环境

- 训练设备 Apple MacbookPro M1pro (Apple Sillion CPU-only)

- python版本

  ```python
   python --version
   Python 3.9.13
  ```

  

- pytorch版本：

  ```python
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

- Open-cv

  ```python
  pip install opencv-python
  ```

- tensorboard

  ```python
  pip install tensorboard
  ```

- Mysql-connector

  ```python
  pip install mysql-connector-python
  ```

  