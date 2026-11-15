#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

class LidarMerger : public rclcpp::Node
{
public:
    LidarMerger() : Node("lidar_merger_node")
    {
        declare_parameter("target_frame", "os_top_sensor");
        target_frame_ = get_parameter("target_frame").as_string();

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // ✅ RELIABLE publisher so RViz & ros2 topic can receive it
        pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            "/points_merged",
            rclcpp::QoS(10).reliable());

        rmw_qos_profile_t qos =
            rclcpp::SensorDataQoS().get_rmw_qos_profile();

        sub_top_.subscribe(this, "/os_top/os_cloud/points", qos);
        sub_front_.subscribe(this, "/os_front/os_cloud/points", qos);
        sub_back_.subscribe(this, "/os_back/os_cloud/points", qos);

        sync_ = std::make_shared<Synchronizer>(
            SyncPolicy(10),
            sub_top_, sub_front_, sub_back_);

        sync_->registerCallback(
            std::bind(&LidarMerger::callback, this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3));

        RCLCPP_INFO(get_logger(),
            "Lidar Merger running. Target frame: %s",
            target_frame_.c_str());
    }

private:
    using PointCloud = pcl::PointCloud<pcl::PointXYZI>;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2,
        sensor_msgs::msg::PointCloud2,
        sensor_msgs::msg::PointCloud2>;
    using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

    void transform_cloud(
        const sensor_msgs::msg::PointCloud2 &msg,
        PointCloud &out)
    {
        pcl::PointCloud<pcl::PointXYZI> cloud;
        pcl::fromROSMsg(msg, cloud);

        if (msg.header.frame_id == target_frame_) {
            out = cloud;
            return;
        }

        auto tf = tf_buffer_->lookupTransform(
            target_frame_,
            msg.header.frame_id,
            tf2::TimePointZero);

        Eigen::Affine3d eigen_tf = tf2::transformToEigen(tf);
        pcl::transformPointCloud(cloud, out, eigen_tf);
    }

    void callback(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr &top,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr &front,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr &back)
    {
        PointCloud cloud_top, cloud_front, cloud_back;

        try {
            transform_cloud(*top, cloud_top);
            transform_cloud(*front, cloud_front);
            transform_cloud(*back, cloud_back);
        } catch (const tf2::TransformException &e) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "TF error: %s", e.what());
            return;
        }

        PointCloud merged;
        merged.reserve(
            cloud_top.size() +
            cloud_front.size() +
            cloud_back.size());

        merged += cloud_top;
        merged += cloud_front;
        merged += cloud_back;

        sensor_msgs::msg::PointCloud2 out;
        pcl::toROSMsg(merged, out);
        out.header.frame_id = target_frame_;
        out.header.stamp = top->header.stamp;

        pub_->publish(out);
    }

    std::string target_frame_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_top_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_front_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_back_;

    std::shared_ptr<Synchronizer> sync_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarMerger>());
    rclcpp::shutdown();
    return 0;
}
