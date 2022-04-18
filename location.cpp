/*******************************************************************************
 * FILENAME:      location.cpp.c
 *
 * AUTHORS:       Mo Yu
 *
 * START DATE:    2022.02.22 11:33:28
 *
 * CONTACT:       yu.mo@smartmore.com
 *******************************************************************************/

#include <smorevision/measurement/intersect_lineline.hpp>
#include <smorevision/core/display.hpp>
#include "location.h"


bool Locations::Train(
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
        const std::vector<std::vector<cv::Point2f>> &mask_rois) {

    gray.copyTo(_template_image.image);
    shape2d::Rect smv_roi(roi.x, roi.y,
                          roi.x + roi.width,
                          roi.y + roi.height);

    std::vector<shape2d::Region2d*> smv_ignore_rois;
    for (const auto &nr: ignore_rois)
    {
        int x1 = nr.x, y1 = nr.y;
        int x2 = nr.x + nr.width, y2 = nr.height;
        shape2d::Region2d* ptr = new shape2d::Rect(x1, y1, x2, y2);
        smv_ignore_rois.emplace_back(ptr);
    }

    _train_params.SetDownSampleFactor(down_sample_factor);
    _train_params.SetAngle(-5, 5, 0.2);
    _run_params.SetNumToFind(search_num);
    _pattern = shapematch::Train(_template_image, _train_params, smv_roi,
                                 smv_ignore_rois);

    SetLineParams(_line1_params, line1_params);
    SetLineParams(_line2_params, line2_params);
    SetLineParams(_line3_params, line3_params);
    SetLineParams(_line4_params, line4_params);

    _mask_contours.clear();
    for (const auto &contours: mask_contours)
    {
        std::vector<cv::Point2f> points;
        for (const auto &p: contours)
        {
            points.emplace_back(p);
        }
        _mask_contours.emplace_back(points);
    }
    _roi.clear();
    for (const auto &contours: mask_rois)
    {
        std::vector<cv::Point2f> points;
        for (const auto &p: contours)
        {
            points.emplace_back(p);
        }
        _roi.emplace_back(points);
    }

    auto out1 = findline::Run(gray, _line1_params);
    if (!out1.GetIsFound())
        return false;
    const shape2d::Line &line1 = out1.GetLine();

    auto out2 = findline::Run(gray, _line2_params);
    if (!out2.GetIsFound())
        return false;
    const shape2d::Line &line2 = out2.GetLine();

    auto out3 = findline::Run(gray, _line3_params);
    if (!out3.GetIsFound())
        return false;
    const shape2d::Line &line3 = out3.GetLine();

    auto out4 = findline::Run(gray, _line4_params);
    if (!out4.GetIsFound())
        return false;
    const shape2d::Line &line4 = out4.GetLine();

    auto p1 = measurement::intersect_lineline::Run(line1, line2);
    if (!p1.IsIntersect())
        return false;

    auto p2 = measurement::intersect_lineline::Run(line2, line3);
    if (!p2.IsIntersect())
        return false;

    auto p3 = measurement::intersect_lineline::Run(line3, line4);
    if (!p3.IsIntersect())
        return false;

    auto p4 = measurement::intersect_lineline::Run(line4, line1);
    if (!p4.IsIntersect())
        return false;

    _rect_points = {{(float)p1.GetIntersectPt().pt.X(), (float)p1.GetIntersectPt().pt.Y()},
                    {(float)p2.GetIntersectPt().pt.X(), (float)p2.GetIntersectPt().pt.Y()},
                    {(float)p3.GetIntersectPt().pt.X(), (float)p3.GetIntersectPt().pt.Y()},
                    {(float)p4.GetIntersectPt().pt.X(), (float)p4.GetIntersectPt().pt.Y()}};

    cx = (double)roi.x + (double)roi.width / 2;
    cy = (double)roi.y + (double)roi.height / 2;

    return true;
}

void Locations::SetLineParams(findline::Params &target,
                              const findline::Params &source) {
    target.SetLineSeg(source.GetLineSeg());
    target.SetNumCalipers(source.GetNumCalipers());
    target.SetCaliperLength(source.GetCaliperLength());
    target.SetCaliperWidth(source.GetCaliperWidth());
    target.SetNumIgnoreEdges(source.GetNumIgnoreEdges());
    target.SetCaliperParams(source.GetCaliperParams());
}

bool Locations::Inference(
        const cv::Mat &image,
        cv::Mat &render,
        std::vector<std::vector<cv::Point2f>> &unit_contours) {
    cv::Mat gray;
    // render = image.clone();
    if (image.channels() == 3)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = image.clone();
    }

    image::Image run_image(gray);
    auto output = shapematch::Run(run_image, _pattern, _run_params);

    std::vector<std::vector<cv::Point>> mask_contours;
    std::vector<std::vector<cv::Point>> unit_mask_contours;
    std::vector<std::vector<cv::Point>> roi_contours;
    for (const auto &result: output.GetMatchResults())
    {

        const auto &pose = result.GetPose();
        auto match_m = pose.GetMatrix();
        match_m(0, 2) -= (match_m(0, 0) * cx + match_m(0, 1) * cy);
        match_m(1, 2) -= (match_m(1, 0) * cx + match_m(1, 1) * cy);

        findline::Params line1_params = _line1_params;
        Map(_line1_params, line1_params, match_m);
        findline::Params line2_params = _line2_params;
        Map(_line2_params, line2_params, match_m);
        findline::Params line3_params = _line3_params;
        Map(_line3_params, line3_params, match_m);
        findline::Params line4_params = _line4_params;
        Map(_line4_params, line4_params, match_m);

        auto out1 = findline::Run(run_image, line1_params);

        if (!out1.GetIsFound())
            return false;
        const shape2d::Line &line1 = out1.GetLine();

        auto out2 = findline::Run(run_image, line2_params);
        if (!out2.GetIsFound())
            return false;
        const shape2d::Line &line2 = out2.GetLine();

        auto out3 = findline::Run(run_image, line3_params);
        if (!out3.GetIsFound())
            return false;
        const shape2d::Line &line3 = out3.GetLine();

        auto out4 = findline::Run(run_image, line4_params);

        if (!out4.GetIsFound())
            return false;
        const shape2d::Line &line4 = out4.GetLine();

//        cv::Mat draw;
//        image.copyTo(draw);
//        result.DrawCVImage(draw, 2);
//        image::Image smv_draw(draw);
//        for (const auto &r: line1_params.GetCaliperRegions())
//        {
//            display::DrawRotatedRect(smv_draw, r, {0, 255, 0}, 2);
//        }
//        for (const auto &r: line2_params.GetCaliperRegions())
//        {
//            display::DrawRotatedRect(smv_draw, r, {0, 255, 0}, 2);
//        }
//        for (const auto &r: line3_params.GetCaliperRegions())
//        {
//            display::DrawRotatedRect(smv_draw, r, {0, 255, 0}, 2);
//        }
//        for (const auto &r: line4_params.GetCaliperRegions())
//        {
//            display::DrawRotatedRect(smv_draw, r, {0, 255, 0}, 2);
//        }
//
//        display::DrawLine(smv_draw, line1, {0, 0, 255}, 2);
//        display::DrawLine(smv_draw, line2, {0, 0, 255}, 2);
//        display::DrawLine(smv_draw, line3, {0, 0, 255}, 2);
//        display::DrawLine(smv_draw, line4, {0, 0, 255}, 2);
//
//        smv_draw.image.copyTo(draw);
//        cv::imwrite("/home/moyu/HeBi-Graphite-Location/draw.png", draw);

        // 4. 求直线交点
        auto p1 = measurement::intersect_lineline::Run(line1, line2);
        if (!p1.IsIntersect())
            return false;

        auto p2 = measurement::intersect_lineline::Run(line2, line3);
        if (!p2.IsIntersect())
            return false;

        auto p3 = measurement::intersect_lineline::Run(line3, line4);
        if (!p3.IsIntersect())
            return false;

        auto p4 = measurement::intersect_lineline::Run(line4, line1);
        if (!p4.IsIntersect())
            return false;

        std::vector<cv::Point2f> inf_rect_points = {
                {(float)p1.GetIntersectPt().pt.X(), (float)p1.GetIntersectPt().pt.Y()},
                {(float)p2.GetIntersectPt().pt.X(), (float)p2.GetIntersectPt().pt.Y()},
                {(float)p3.GetIntersectPt().pt.X(), (float)p3.GetIntersectPt().pt.Y()},
                {(float)p4.GetIntersectPt().pt.X(), (float)p4.GetIntersectPt().pt.Y()}};
        unit_contours.emplace_back(inf_rect_points);
        cv::Mat m = cv::getPerspectiveTransform(_rect_points, inf_rect_points);

        for (const auto &contours: _mask_contours)
        {
            std::vector<cv::Point> points;
            for (const auto &p: contours)
            {
                cv::Point2d pd = cv::Point2d(p.x, p.y);
                cv::Point2d npd;
                Map(pd, npd, m);
                points.emplace_back(cv::Point((int)npd.x, (int)npd.y));
            }
            mask_contours.emplace_back(points);
        }

        for (const auto &contours: _roi)
        {
            std::vector<cv::Point> points;
            for (const auto &p: contours)
            {
                cv::Point2d pd = cv::Point2d(p.x, p.y);
                cv::Point2d npd;
                Map(pd, npd, m);
                points.emplace_back(cv::Point((int)npd.x, (int)npd.y));
            }
            roi_contours.emplace_back(points);
        }

    }

    image.copyTo(render);
    for (int i = 0; i < mask_contours.size(); ++i)
    {
        cv::drawContours(render, mask_contours, i, cv::Vec3b(0, 0, 0), -1);
    }
    if (roi_contours.empty())
        return true;

    cv::Mat roi_image = cv::Mat::ones(render.size(), CV_8U) * 255;
    for (int i = 0; i < roi_contours.size(); ++i)
    {
        cv::drawContours(roi_image, roi_contours, i, 0, -1);
    }
    render.setTo(cv::Vec3b(0, 0, 0), roi_image);

    return true;
}


void Locations::Map(const math::Vector2d &source_point,
                    math::Vector2d &target_point,
                    const math::Matrix3d &m) {
    double x = source_point.X(), y = source_point.Y();
    auto u = (m(0, 0) * x + m(0, 1) * y + m(0, 2)) / (m(2, 0) * x + m(2, 1) * y + m(2, 2));
    auto v = (m(1, 0) * x + m(1, 1) * y + m(1, 2)) / (m(2, 0) * x + m(2, 1) * y + m(2, 2));
    target_point.X() = u;
    target_point.Y() = v;
}

void Locations::Map(const cv::Point2d &source_point,
                    cv::Point2d &target_point,
                    const cv::Mat &m) {
    double x = source_point.x, y = source_point.y;
    auto m00 = (float)m.at<double>(0, 0);
    auto m01 = (float)m.at<double>(0, 1);
    auto m02 = (float)m.at<double>(0, 2);
    auto m10 = (float)m.at<double>(1, 0);
    auto m11 = (float)m.at<double>(1, 1);
    auto m12 = (float)m.at<double>(1, 2);
    auto m20 = (float)m.at<double>(2, 0);
    auto m21 = (float)m.at<double>(2, 1);
    auto m22 = (float)m.at<double>(2, 2);
    auto u = (m00 * x + m01 * y + m02) / (m20 * x + m21 * y + m22);
    auto v = (m10 * x + m11 * y + m12) / (m20 * x + m21 * y + m22);
    target_point = {u, v};
}

void Locations::Map(const shape2d::LineSegment &source_seg,
                    shape2d::LineSegment &target_seg,
                    const math::Matrix3d &m) {
    Map(source_seg.p1, target_seg.p1, m);
    Map(source_seg.p2, target_seg.p2, m);
}

void Locations::Map(const findline::Params &source_params,
                    findline::Params &target_params,
                    const math::Matrix3d &m) {
    target_params = source_params;
    shape2d::LineSegment seg;
    Map(target_params.GetLineSeg(), seg, m);
    target_params.SetLineSeg(seg);
}
