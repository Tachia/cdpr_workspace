#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gazebo/common/Events.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <xmlrpcpp/XmlRpcValue.h>

namespace gazebo
{
namespace
{
constexpr double kEps = 1e-9;

double Clamp(const double value, const double lo, const double hi)
{
  return std::max(lo, std::min(hi, value));
}

ignition::math::Vector3d XmlVector(const XmlRpc::XmlRpcValue &value)
{
  if (value.getType() != XmlRpc::XmlRpcValue::TypeArray || value.size() != 3)
  {
    return ignition::math::Vector3d::Zero;
  }
  return ignition::math::Vector3d(
      static_cast<double>(value[0]),
      static_cast<double>(value[1]),
      static_cast<double>(value[2]));
}

std::vector<ignition::math::Vector3d> DefaultAnchors()
{
  return {
      {-6.0, -6.0, 6.0}, {6.0, -6.0, 6.0}, {6.0, 6.0, 6.0}, {-6.0, 6.0, 6.0},
      {-6.0, -6.0, 0.0}, {6.0, -6.0, 0.0}, {6.0, 6.0, 0.0}, {-6.0, 6.0, 0.0}};
}

std::vector<ignition::math::Vector3d> DefaultAttachments()
{
  return {
      {-0.5, -0.5, 0.5}, {0.5, -0.5, 0.5}, {0.5, 0.5, 0.5}, {-0.5, 0.5, 0.5},
      {-0.5, -0.5, -0.5}, {0.5, -0.5, -0.5}, {0.5, 0.5, -0.5}, {-0.5, 0.5, -0.5}};
}
}  // namespace

class CdprForcePlugin : public ModelPlugin
{
public:
  CdprForcePlugin() = default;
  ~CdprForcePlugin() override
  {
    queue_.clear();
    queue_.disable();
    if (ros_queue_thread_.joinable())
    {
      ros_queue_thread_.join();
    }
  }

  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = model;
    world_ = model_->GetWorld();

    std::string link_name = "platform_link";
    if (sdf->HasElement("linkName"))
    {
      link_name = sdf->Get<std::string>("linkName");
    }
    link_ = model_->GetLink(link_name);
    if (!link_)
    {
      gzerr << "[cdpr_force_plugin] Link '" << link_name << "' not found.\n";
      return;
    }

    if (!ros::isInitialized())
    {
      int argc = 0;
      char **argv = nullptr;
      ros::init(argc, argv, "cdpr_force_plugin", ros::init_options::NoSigintHandler);
    }

    ros_node_.reset(new ros::NodeHandle(""));
    ros_node_->setCallbackQueue(&queue_);
    LoadParameters();

    tension_sub_ = ros_node_->subscribe("/cable_tensions", 1, &CdprForcePlugin::OnTensionCommand, this);
    actual_pub_ = ros_node_->advertise<std_msgs::Float32MultiArray>("/actual_cable_tensions", 1);
    sag_pub_ = ros_node_->advertise<std_msgs::Float32MultiArray>("/sag_directions", 1);

    update_connection_ = event::Events::ConnectWorldUpdateBegin(
        std::bind(&CdprForcePlugin::OnUpdate, this));
    ros_queue_thread_ = std::thread(std::bind(&CdprForcePlugin::QueueThread, this));

    gzmsg << "[cdpr_force_plugin] Loaded for " << anchors_.size()
          << " cables, model_type=" << model_type_ << "\n";
  }

private:
  struct CableResult
  {
    ignition::math::Vector3d unit = ignition::math::Vector3d::Zero;
    double platform_tension = 0.0;
    double sag = 0.0;
  };

  void LoadParameters()
  {
    model_type_ = 0;
    ros_node_->param("/cdpr/model_type", model_type_, model_type_);
    ros_node_->param("/cdpr/cable/min_tension", min_tension_, 10.0);
    ros_node_->param("/cdpr/cable/max_tension", max_tension_, 500.0);
    ros_node_->param("/cdpr/cable/nominal_tension", nominal_tension_, 80.0);
    ros_node_->param("/cdpr/cable/linear_density", linear_density_, 0.00739);
    ros_node_->param("/cdpr/cable/axial_stiffness", axial_stiffness_, 7.06858347e4);
    ros_node_->param("/cdpr/cable/gravity", gravity_, 9.80665);

    anchors_ = DefaultAnchors();
    attachments_ = DefaultAttachments();

    XmlRpc::XmlRpcValue anchors_xml;
    if (ros_node_->getParam("/cdpr/anchors", anchors_xml) &&
        anchors_xml.getType() == XmlRpc::XmlRpcValue::TypeArray)
    {
      anchors_.clear();
      for (int i = 0; i < anchors_xml.size(); ++i)
      {
        anchors_.push_back(XmlVector(anchors_xml[i]));
      }
    }

    XmlRpc::XmlRpcValue attachments_xml;
    if (ros_node_->getParam("/cdpr/platform_attachments", attachments_xml) &&
        attachments_xml.getType() == XmlRpc::XmlRpcValue::TypeArray)
    {
      attachments_.clear();
      for (int i = 0; i < attachments_xml.size(); ++i)
      {
        attachments_.push_back(XmlVector(attachments_xml[i]));
      }
    }

    const std::size_t n = std::min(anchors_.size(), attachments_.size());
    anchors_.resize(n);
    attachments_.resize(n);
    commanded_tensions_.assign(n, nominal_tension_);
  }

  void OnTensionCommand(const std_msgs::Float32MultiArrayConstPtr &msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t n = commanded_tensions_.size();
    for (std::size_t i = 0; i < n && i < msg->data.size(); ++i)
    {
      commanded_tensions_[i] = Clamp(static_cast<double>(msg->data[i]), 0.0, max_tension_);
    }
  }

  CableResult StraightCable(const ignition::math::Vector3d &anchor,
                            const ignition::math::Vector3d &point,
                            const double command) const
  {
    CableResult result;
    const ignition::math::Vector3d delta = anchor - point;
    const double length = delta.Length();
    if (length > kEps)
    {
      result.unit = delta / length;
    }
    result.platform_tension = Clamp(command, 0.0, max_tension_);
    return result;
  }

  double EndpointTensionForH(const double horizontal_tension,
                             const double horizontal_span,
                             const double vertical_delta,
                             double *slope0,
                             double *sag_mid) const
  {
    const double w = std::max(linear_density_ * gravity_, 1e-9);
    const double a = std::max(horizontal_tension / w, 1e-9);
    const double half_arg = horizontal_span / (2.0 * a);
    if (std::abs(half_arg) > 50.0)
    {
      if (slope0)
      {
        *slope0 = 0.0;
      }
      if (sag_mid)
      {
        *sag_mid = std::numeric_limits<double>::infinity();
      }
      return std::numeric_limits<double>::infinity();
    }
    const double denom = 2.0 * a * std::sinh(half_arg);
    const double c = horizontal_span / 2.0 - a * std::asinh(vertical_delta / std::max(denom, 1e-12));
    const double slope = std::sinh(-c / a);
    if (slope0)
    {
      *slope0 = slope;
    }
    if (sag_mid)
    {
      const double z_mid = a * (std::cosh((horizontal_span / 2.0 - c) / a) - std::cosh((-c) / a));
      const double z_line = vertical_delta * 0.5;
      *sag_mid = z_line - z_mid;
    }
    return horizontal_tension * std::sqrt(1.0 + slope * slope);
  }

  double SolveHorizontalTension(const double platform_tension,
                                const double horizontal_span,
                                const double vertical_delta) const
  {
    if (horizontal_span < 1e-6)
    {
      return platform_tension;
    }

    auto f = [&](double h) {
      return EndpointTensionForH(h, horizontal_span, vertical_delta, nullptr, nullptr) - platform_tension;
    };

    double lo = std::max(1e-3, platform_tension * 0.02);
    double hi = std::max(1.0, platform_tension * 1.5);
    double flo = f(lo);
    double fhi = f(hi);

    int expand = 0;
    while (flo > 0.0 && expand++ < 30)
    {
      lo *= 0.5;
      flo = f(lo);
    }
    expand = 0;
    while (fhi < 0.0 && expand++ < 30)
    {
      hi *= 2.0;
      fhi = f(hi);
    }

    if (flo * fhi > 0.0)
    {
      const double line_slope = vertical_delta / std::max(horizontal_span, 1e-9);
      return platform_tension / std::sqrt(1.0 + line_slope * line_slope);
    }

    for (int iter = 0; iter < 60; ++iter)
    {
      const double mid = 0.5 * (lo + hi);
      const double fmid = f(mid);
      if (fmid > 0.0)
      {
        hi = mid;
      }
      else
      {
        lo = mid;
      }
    }
    return 0.5 * (lo + hi);
  }

  CableResult CatenaryCable(const ignition::math::Vector3d &anchor,
                            const ignition::math::Vector3d &point,
                            const double command) const
  {
    const ignition::math::Vector3d delta = anchor - point;
    const double length = std::max(delta.Length(), 1e-9);
    const double cable_weight = linear_density_ * gravity_ * length;
    const double platform_tension = Clamp(command - 0.5 * cable_weight, 0.0, max_tension_);

    if (platform_tension <= 1e-6 || length <= 1e-6)
    {
      return CableResult{};
    }

    ignition::math::Vector3d horizontal(delta.X(), delta.Y(), 0.0);
    const double horizontal_span = horizontal.Length();
    if (horizontal_span <= 1e-6)
    {
      return StraightCable(anchor, point, platform_tension);
    }

    const double vertical_delta = delta.Z();
    const ignition::math::Vector3d eh = horizontal / horizontal_span;
    const double h = SolveHorizontalTension(platform_tension, horizontal_span, vertical_delta);
    double slope0 = vertical_delta / horizontal_span;
    double sag_mid = 0.0;
    EndpointTensionForH(h, horizontal_span, vertical_delta, &slope0, &sag_mid);

    ignition::math::Vector3d tangent(eh.X(), eh.Y(), slope0);
    tangent.Normalize();

    CableResult result;
    result.unit = tangent;
    result.platform_tension = platform_tension;
    result.sag = sag_mid;
    return result;
  }

  void OnUpdate()
  {
    if (!link_)
    {
      return;
    }

    std::vector<double> commands;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      commands = commanded_tensions_;
    }

    const ignition::math::Pose3d pose = link_->WorldPose();
    std::vector<double> actual(commands.size(), 0.0);
    std::vector<ignition::math::Vector3d> sag_dirs(commands.size(), ignition::math::Vector3d::Zero);

    for (std::size_t i = 0; i < commands.size(); ++i)
    {
      const ignition::math::Vector3d local = attachments_[i];
      const ignition::math::Vector3d world_point = pose.Pos() + pose.Rot().RotateVector(local);
      CableResult cable = model_type_ == 1
                               ? CatenaryCable(anchors_[i], world_point, commands[i])
                               : StraightCable(anchors_[i], world_point, commands[i]);

      const ignition::math::Vector3d force = cable.unit * cable.platform_tension;
      link_->AddForceAtWorldPosition(force, world_point);
      actual[i] = cable.platform_tension;
      sag_dirs[i] = cable.unit;
    }

    Publish(actual, sag_dirs);
  }

  void Publish(const std::vector<double> &actual,
               const std::vector<ignition::math::Vector3d> &directions)
  {
    if (actual_pub_.getNumSubscribers() > 0)
    {
      std_msgs::Float32MultiArray msg;
      msg.data.reserve(actual.size());
      for (double t : actual)
      {
        msg.data.push_back(static_cast<float>(t));
      }
      actual_pub_.publish(msg);
    }

    if (model_type_ == 1 && sag_pub_.getNumSubscribers() > 0)
    {
      std_msgs::Float32MultiArray msg;
      msg.layout.dim.resize(2);
      msg.layout.dim[0].label = "cable";
      msg.layout.dim[0].size = directions.size();
      msg.layout.dim[0].stride = directions.size() * 3;
      msg.layout.dim[1].label = "xyz";
      msg.layout.dim[1].size = 3;
      msg.layout.dim[1].stride = 3;
      msg.data.reserve(directions.size() * 3);
      for (const auto &u : directions)
      {
        msg.data.push_back(static_cast<float>(u.X()));
        msg.data.push_back(static_cast<float>(u.Y()));
        msg.data.push_back(static_cast<float>(u.Z()));
      }
      sag_pub_.publish(msg);
    }
  }

  void QueueThread()
  {
    static const double timeout = 0.01;
    while (ros_node_ && ros_node_->ok())
    {
      queue_.callAvailable(ros::WallDuration(timeout));
    }
  }

  physics::ModelPtr model_;
  physics::WorldPtr world_;
  physics::LinkPtr link_;
  event::ConnectionPtr update_connection_;

  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;
  ros::Subscriber tension_sub_;
  ros::Publisher actual_pub_;
  ros::Publisher sag_pub_;

  std::mutex mutex_;
  std::vector<ignition::math::Vector3d> anchors_;
  std::vector<ignition::math::Vector3d> attachments_;
  std::vector<double> commanded_tensions_;

  int model_type_ = 0;
  double min_tension_ = 10.0;
  double max_tension_ = 500.0;
  double nominal_tension_ = 80.0;
  double linear_density_ = 0.00739;
  double axial_stiffness_ = 7.06858347e4;
  double gravity_ = 9.80665;
};

GZ_REGISTER_MODEL_PLUGIN(CdprForcePlugin)
}  // namespace gazebo
