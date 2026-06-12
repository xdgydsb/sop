package com.ruoyi.system.domain.sop;

import java.math.BigDecimal;
import java.util.Date;
import com.fasterxml.jackson.annotation.JsonFormat;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;
import com.ruoyi.common.annotation.Excel;
import com.ruoyi.common.core.domain.BaseEntity;

/**
 * SOP detection event log object sop_detection_event
 *
 * @author ruoyi
 */
public class SopDetectionEvent extends BaseEntity
{
    private static final long serialVersionUID = 1L;

    /** Event log ID */
    private Long eventLogId;

    /** External request ID */
    @Excel(name = "请求ID")
    private String requestId;

    /** Task ID */
    @Excel(name = "任务ID")
    private Long taskId;

    /** Task code */
    @Excel(name = "任务编码")
    private String taskCode;

    /** Product code */
    @Excel(name = "产品编码")
    private String productCode;

    /** Station code */
    @Excel(name = "工位编码")
    private String stationCode;

    /** Camera code */
    @Excel(name = "相机编码")
    private String cameraCode;

    /** External event ID */
    @Excel(name = "事件ID")
    private String eventId;

    /** Event code */
    @Excel(name = "事件编码")
    private String eventCode;

    /** Event name */
    @Excel(name = "事件名称")
    private String eventName;

    /** Confidence */
    @Excel(name = "置信度")
    private BigDecimal confidence;

    /** External event time */
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @Excel(name = "事件时间", width = 30, dateFormat = "yyyy-MM-dd HH:mm:ss")
    private Date eventTime;

    /** Receive time */
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @Excel(name = "接收时间", width = 30, dateFormat = "yyyy-MM-dd HH:mm:ss")
    private Date receiveTime;

    /** Evidence image URL */
    @Excel(name = "图片地址")
    private String imageUrl;

    /** Raw event payload */
    private String rawPayload;

    /** Judge result */
    @Excel(name = "判定结果")
    private String judgeResult;

    /** Judge message */
    @Excel(name = "判定说明")
    private String judgeMessage;

    /** Related step ID */
    @Excel(name = "步骤ID")
    private Long stepId;

    /** Related step number */
    @Excel(name = "步骤序号")
    private Integer stepNo;

    public Long getEventLogId()
    {
        return eventLogId;
    }

    public void setEventLogId(Long eventLogId)
    {
        this.eventLogId = eventLogId;
    }

    public String getRequestId()
    {
        return requestId;
    }

    public void setRequestId(String requestId)
    {
        this.requestId = requestId;
    }

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

    public String getProductCode()
    {
        return productCode;
    }

    public void setProductCode(String productCode)
    {
        this.productCode = productCode;
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

    public String getEventId()
    {
        return eventId;
    }

    public void setEventId(String eventId)
    {
        this.eventId = eventId;
    }

    public String getEventCode()
    {
        return eventCode;
    }

    public void setEventCode(String eventCode)
    {
        this.eventCode = eventCode;
    }

    public String getEventName()
    {
        return eventName;
    }

    public void setEventName(String eventName)
    {
        this.eventName = eventName;
    }

    public BigDecimal getConfidence()
    {
        return confidence;
    }

    public void setConfidence(BigDecimal confidence)
    {
        this.confidence = confidence;
    }

    public Date getEventTime()
    {
        return eventTime;
    }

    public void setEventTime(Date eventTime)
    {
        this.eventTime = eventTime;
    }

    public Date getReceiveTime()
    {
        return receiveTime;
    }

    public void setReceiveTime(Date receiveTime)
    {
        this.receiveTime = receiveTime;
    }

    public String getImageUrl()
    {
        return imageUrl;
    }

    public void setImageUrl(String imageUrl)
    {
        this.imageUrl = imageUrl;
    }

    public String getRawPayload()
    {
        return rawPayload;
    }

    public void setRawPayload(String rawPayload)
    {
        this.rawPayload = rawPayload;
    }

    public String getJudgeResult()
    {
        return judgeResult;
    }

    public void setJudgeResult(String judgeResult)
    {
        this.judgeResult = judgeResult;
    }

    public String getJudgeMessage()
    {
        return judgeMessage;
    }

    public void setJudgeMessage(String judgeMessage)
    {
        this.judgeMessage = judgeMessage;
    }

    public Long getStepId()
    {
        return stepId;
    }

    public void setStepId(Long stepId)
    {
        this.stepId = stepId;
    }

    public Integer getStepNo()
    {
        return stepNo;
    }

    public void setStepNo(Integer stepNo)
    {
        this.stepNo = stepNo;
    }

    @Override
    public String toString()
    {
        return new ToStringBuilder(this, ToStringStyle.MULTI_LINE_STYLE)
            .append("eventLogId", getEventLogId())
            .append("requestId", getRequestId())
            .append("taskId", getTaskId())
            .append("taskCode", getTaskCode())
            .append("productCode", getProductCode())
            .append("stationCode", getStationCode())
            .append("cameraCode", getCameraCode())
            .append("eventId", getEventId())
            .append("eventCode", getEventCode())
            .append("eventName", getEventName())
            .append("confidence", getConfidence())
            .append("eventTime", getEventTime())
            .append("receiveTime", getReceiveTime())
            .append("imageUrl", getImageUrl())
            .append("rawPayload", getRawPayload())
            .append("judgeResult", getJudgeResult())
            .append("judgeMessage", getJudgeMessage())
            .append("stepId", getStepId())
            .append("stepNo", getStepNo())
            .append("createBy", getCreateBy())
            .append("createTime", getCreateTime())
            .append("updateBy", getUpdateBy())
            .append("updateTime", getUpdateTime())
            .append("remark", getRemark())
            .toString();
    }
}
