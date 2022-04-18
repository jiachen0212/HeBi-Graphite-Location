/*******************************************************************************
 * FILENAME:      location.h
 *
 * AUTHORS:       Mo Yu
 *
 * START DATE:    2022.02.22 11:33:28
 *
 * CONTACT:       yu.mo@smartmore.com
 *******************************************************************************/

#ifndef EXAMPLE_LOCATION_H
#define EXAMPLE_LOCATION_H


#include <vector>

#include <smorevision/detection/findline.hpp>
#include <smorevision/matching/shapematch.hpp>

using namespace smorevision;
using namespace smorevision::detection;
using namespace smorevision::matching;


class Locations{
public:
    Locations() = default;
    ~Locations() = default;
    bool Train(
            const cv::Mat &gray,
            const cv::Rect &roi,
            const std::vector<cv::Rect> &ignore_rois,
            int search_num,
            int down_sample_factor,
            const findline::Params &line1_params,
            const findline::Params &line2_params,
            const findline::Params &line3_params,
            const findline::Params &line4_params,
            const std::vector<std::vector<cv::Point2f>> &mask_contours,
            const std::vector<std::vector<cv::Point2f>> &mask_rois);
    bool Inference(
            const cv::Mat &image,
            cv::Mat &render,
            std::vector<std::vector<cv::Point2f>> &unit_contours);

private:
    static void SetLineParams(findline::Params &target,
                              const findline::Params &source);
    static void Map(const math::Vector2d &source_point,
                    math::Vector2d &target_point,
                    const math::Matrix3d &m);
    static void Map(const shape2d::LineSegment &source_seg,
                    shape2d::LineSegment &target_seg,
                    const math::Matrix3d &m);
    static void Map(const findline::Params &source_params,
                    findline::Params &target_params,
                    const math::Matrix3d &m);
    static void Map(const cv::Point2d &source_point,
                    cv::Point2d &target_point,
                    const cv::Mat &m);

private:
    findline::Params _line1_params;
    findline::Params _line2_params;
    findline::Params _line3_params;
    findline::Params _line4_params;
    std::vector<cv::Point2f> _rect_points;
    std::vector<std::vector<cv::Point2f>> _mask_contours;
    std::vector<std::vector<cv::Point2f>> _roi;
    image::Image _template_image;
    shapematch::TrainParams _train_params;
    shapematch::RunParams _run_params;
    shapematch::ShapePattern _pattern;
    double cx = 0;
    double cy = 0;
};

#endif //EXAMPLE_LOCATION_H
