package com.ruoyi.system.domain.sop;

import java.math.BigDecimal;
import java.util.Date;
import com.fasterxml.jackson.annotation.JsonFormat;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;
import com.ruoyi.common.annotation.Excel;
import com.ruoyi.common.core.domain.BaseEntity;

/**
 * SOP task step snapshot object sop_task_step
 *
 * @author ruoyi
 */
public class SopTaskStep extends BaseEntity
{
    private static final long serialVersionUID = 1L;

    /** Task step ID */
    private Long taskStepId;

    /** Task ID */
    @Excel(name = "任务ID")
    private Long taskId;

    /** SOP step ID */
    @Excel(name = "步骤ID")
    private Long stepId;

    /** Step sequence number */
    @Excel(name = "步骤序号")
    private Integer stepNo;

    /** Step name snapshot */
    @Excel(name = "步骤名称")
    private String stepName;

    /** Expected event code snapshot */
    @Excel(name = "期望事件编码")
    private String expectedEvent;

    /** Required confidence snapshot */
    @Excel(name = "最低置信度")
    private BigDecimal requiredConfidence;

    /** Step status */
    @Excel(name = "步骤状态")
    private String stepStatus;

    /** Pass time */
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @Excel(name = "通过时间", width = 30, dateFormat = "yyyy-MM-dd HH:mm:ss")
    private Date passTime;

    /** Matched event log ID */
    @Excel(name = "事件日志ID")
    private Long eventLogId;

    private String snapshotUrl;

    private String clipUrl;

    private Long clipStartMs;

    private Long clipEndMs;

    private String judgeResult;

    private String judgeMessage;

    public Long getTaskStepId()
    {
        return taskStepId;
    }

    public void setTaskStepId(Long taskStepId)
    {
        this.taskStepId = taskStepId;
    }

    public Long getTaskId()
    {
        return taskId;
    }

    public void setTaskId(Long taskId)
    {
        this.taskId = taskId;
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

    public String getStepName()
    {
        return stepName;
    }

    public void setStepName(String stepName)
    {
        this.stepName = stepName;
    }

    public String getExpectedEvent()
    {
        return expectedEvent;
    }

    public void setExpectedEvent(String expectedEvent)
    {
        this.expectedEvent = expectedEvent;
    }

    public BigDecimal getRequiredConfidence()
    {
        return requiredConfidence;
    }

    public void setRequiredConfidence(BigDecimal requiredConfidence)
    {
        this.requiredConfidence = requiredConfidence;
    }

    public String getStepStatus()
    {
        return stepStatus;
    }

    public void setStepStatus(String stepStatus)
    {
        this.stepStatus = stepStatus;
    }

    public Date getPassTime()
    {
        return passTime;
    }

    public void setPassTime(Date passTime)
    {
        this.passTime = passTime;
    }

    public Long getEventLogId()
    {
        return eventLogId;
    }

    public void setEventLogId(Long eventLogId)
    {
        this.eventLogId = eventLogId;
    }

    public String getSnapshotUrl()
    {
        return snapshotUrl;
    }

    public void setSnapshotUrl(String snapshotUrl)
    {
        this.snapshotUrl = snapshotUrl;
    }

    public String getClipUrl()
    {
        return clipUrl;
    }

    public void setClipUrl(String clipUrl)
    {
        this.clipUrl = clipUrl;
    }

    public Long getClipStartMs()
    {
        return clipStartMs;
    }

    public void setClipStartMs(Long clipStartMs)
    {
        this.clipStartMs = clipStartMs;
    }

    public Long getClipEndMs()
    {
        return clipEndMs;
    }

    public void setClipEndMs(Long clipEndMs)
    {
        this.clipEndMs = clipEndMs;
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

    @Override
    public String toString()
    {
        return new ToStringBuilder(this, ToStringStyle.MULTI_LINE_STYLE)
            .append("taskStepId", getTaskStepId())
            .append("taskId", getTaskId())
            .append("stepId", getStepId())
            .append("stepNo", getStepNo())
            .append("stepName", getStepName())
            .append("expectedEvent", getExpectedEvent())
            .append("requiredConfidence", getRequiredConfidence())
            .append("stepStatus", getStepStatus())
            .append("passTime", getPassTime())
            .append("eventLogId", getEventLogId())
            .append("snapshotUrl", getSnapshotUrl())
            .append("clipUrl", getClipUrl())
            .append("clipStartMs", getClipStartMs())
            .append("clipEndMs", getClipEndMs())
            .append("judgeResult", getJudgeResult())
            .append("judgeMessage", getJudgeMessage())
            .append("createBy", getCreateBy())
            .append("createTime", getCreateTime())
            .append("updateBy", getUpdateBy())
            .append("updateTime", getUpdateTime())
            .append("remark", getRemark())
            .toString();
    }
}
