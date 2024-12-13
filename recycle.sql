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
