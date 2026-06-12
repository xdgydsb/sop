package com.ruoyi.system.domain.sop;

import java.util.Date;
import com.fasterxml.jackson.annotation.JsonFormat;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;
import com.ruoyi.common.annotation.Excel;
import com.ruoyi.common.core.domain.BaseEntity;

/**
 * SOP detection task object sop_detection_task
 *
 * @author ruoyi
 */
public class SopDetectionTask extends BaseEntity
{
    private static final long serialVersionUID = 1L;

    /** Task ID */
    private Long taskId;

    /** Task code */
    @Excel(name = "任务编码")
    private String taskCode;

    /** Product ID */
    @Excel(name = "产品ID")
    private Long productId;

    /** Product code */
    @Excel(name = "产品编码")
    private String productCode;

    /** Product name */
    @Excel(name = "产品名称")
    private String productName;

    /** SOP ID */
    @Excel(name = "SOP ID")
    private Long sopId;

    /** SOP name */
    @Excel(name = "SOP名称")
    private String sopName;

    /** Station code */
    @Excel(name = "工位编码")
    private String stationCode;

    /** Camera code */
    @Excel(name = "相机编码")
    private String cameraCode;

    /** Current step number */
    @Excel(name = "当前步骤")
    private Integer currentStepNo;

    /** Task status */
    @Excel(name = "任务状态")
    private String taskStatus;

    /** Start time */
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @Excel(name = "开始时间", width = 30, dateFormat = "yyyy-MM-dd HH:mm:ss")
    private Date startTime;

    /** End time */
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @Excel(name = "结束时间", width = 30, dateFormat = "yyyy-MM-dd HH:mm:ss")
    private Date endTime;

    /** Operator name */
    @Excel(name = "操作员")
    private String operatorName;

    private String previewStreamUrl;

    private String latestFrameUrl;

    private String runtimeMode;

    private String runtimeMessage;

    private Double runtimeFps;

    public Long getTaskId()
    {
        return taskId;
    }

    public void setTaskId(Long taskId)
    {
        this.taskId = taskId;
    }

    public String getTaskCode()
    {
        return taskCode;
    }

    public void setTaskCode(String taskCode)
    {
        this.taskCode = taskCode;
    }

    public Long getProductId()
    {
        return productId;
    }

    public void setProductId(Long productId)
    {
        this.productId = productId;
    }

    public String getProductCode()
    {
        return productCode;
    }

    public void setProductCode(String productCode)
    {
        this.productCode = productCode;
    }

    public String getProductName()
    {
        return productName;
    }

    public void setProductName(String productName)
    {
        this.productName = productName;
    }

    public Long getSopId()
    {
        return sopId;
    }

    public void setSopId(Long sopId)
    {
        this.sopId = sopId;
    }

    public String getSopName()
    {
        return sopName;
    }

    public void setSopName(String sopName)
    {
        this.sopName = sopName;
    }

    public String getStationCode()
    {
        return stationCode;
    }

    public void setStationCode(String stationCode)
    {
        this.stationCode = stationCode;
    }

    public String getCameraCode()
    {
        return cameraCode;
    }

    public void setCameraCode(String cameraCode)
    {
        this.cameraCode = cameraCode;
    }

    public Integer getCurrentStepNo()
    {
        return currentStepNo;
    }

    public void setCurrentStepNo(Integer currentStepNo)
    {
        this.currentStepNo = currentStepNo;
    }

    public String getTaskStatus()
    {
        return taskStatus;
    }

    public void setTaskStatus(String taskStatus)
    {
        this.taskStatus = taskStatus;
    }

    public Date getStartTime()
    {
        return startTime;
    }

    public void setStartTime(Date startTime)
    {
        this.startTime = startTime;
    }

    public Date getEndTime()
    {
        return endTime;
    }

    public void setEndTime(Date endTime)
    {
        this.endTime = endTime;
    }

    public String getOperatorName()
    {
        return operatorName;
    }

    public void setOperatorName(String operatorName)
    {
        this.operatorName = operatorName;
    }

    public String getPreviewStreamUrl()
    {
        return previewStreamUrl;
    }

    public void setPreviewStreamUrl(String previewStreamUrl)
    {
        this.previewStreamUrl = previewStreamUrl;
    }

    public String getLatestFrameUrl()
    {
        return latestFrameUrl;
    }

    public void setLatestFrameUrl(String latestFrameUrl)
    {
        this.latestFrameUrl = latestFrameUrl;
    }

    public String getRuntimeMode()
    {
        return runtimeMode;
    }

    public void setRuntimeMode(String runtimeMode)
    {
        this.runtimeMode = runtimeMode;
    }

    public String getRuntimeMessage()
    {
        return runtimeMessage;
    }

    public void setRuntimeMessage(String runtimeMessage)
    {
        this.runtimeMessage = runtimeMessage;
    }

    public Double getRuntimeFps()
    {
        return runtimeFps;
    }

    public void setRuntimeFps(Double runtimeFps)
    {
        this.runtimeFps = runtimeFps;
    }

    @Override
    public String toString()
    {
        return new ToStringBuilder(this, ToStringStyle.MULTI_LINE_STYLE)
            .append("taskId", getTaskId())
            .append("taskCode", getTaskCode())
            .append("productId", getProductId())
            .append("productCode", getProductCode())
            .append("productName", getProductName())
            .append("sopId", getSopId())
            .append("sopName", getSopName())
            .append("stationCode", getStationCode())
            .append("cameraCode", getCameraCode())
            .append("currentStepNo", getCurrentStepNo())
            .append("taskStatus", getTaskStatus())
            .append("startTime", getStartTime())
            .append("endTime", getEndTime())
            .append("operatorName", getOperatorName())
            .append("previewStreamUrl", getPreviewStreamUrl())
            .append("latestFrameUrl", getLatestFrameUrl())
            .append("runtimeMode", getRuntimeMode())
            .append("runtimeMessage", getRuntimeMessage())
            .append("runtimeFps", getRuntimeFps())
            .append("createBy", getCreateBy())
            .append("createTime", getCreateTime())
            .append("updateBy", getUpdateBy())
            .append("updateTime", getUpdateTime())
            .append("remark", getRemark())
            .toString();
    }
}
