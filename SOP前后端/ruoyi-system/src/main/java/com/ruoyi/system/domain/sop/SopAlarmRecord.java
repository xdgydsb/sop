package com.ruoyi.system.domain.sop;

import java.util.Date;
import com.fasterxml.jackson.annotation.JsonFormat;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;
import com.ruoyi.common.annotation.Excel;
import com.ruoyi.common.core.domain.BaseEntity;

/**
 * SOP alarm record object sop_alarm_record
 *
 * @author ruoyi
 */
public class SopAlarmRecord extends BaseEntity
{
    private static final long serialVersionUID = 1L;

    /** Alarm ID */
    private Long alarmId;

    /** Alarm code */
    @Excel(name = "告警编码")
    private String alarmCode;

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

    /** Alarm type */
    @Excel(name = "告警类型")
    private String alarmType;

    /** Alarm level */
    @Excel(name = "告警级别")
    private String alarmLevel;

    /** Alarm message */
    @Excel(name = "告警内容")
    private String alarmMessage;

    /** Event log ID */
    @Excel(name = "事件日志ID")
    private Long eventLogId;

    /** Event code */
    @Excel(name = "事件编码")
    private String eventCode;

    /** Event name */
    @Excel(name = "事件名称")
    private String eventName;

    /** Related step ID */
    @Excel(name = "步骤ID")
    private Long stepId;

    /** Related step number */
    @Excel(name = "步骤序号")
    private Integer stepNo;

    /** Alarm time */
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @Excel(name = "告警时间", width = 30, dateFormat = "yyyy-MM-dd HH:mm:ss")
    private Date alarmTime;

    /** Handle status */
    @Excel(name = "处理状态")
    private String handleStatus;

    /** Handle by */
    @Excel(name = "处理人")
    private String handleBy;

    /** Handle time */
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @Excel(name = "处理时间", width = 30, dateFormat = "yyyy-MM-dd HH:mm:ss")
    private Date handleTime;

    /** Handle remark */
    @Excel(name = "处理说明")
    private String handleRemark;

    public Long getAlarmId()
    {
        return alarmId;
    }

    public void setAlarmId(Long alarmId)
    {
        this.alarmId = alarmId;
    }

    public String getAlarmCode()
    {
        return alarmCode;
    }

    public void setAlarmCode(String alarmCode)
    {
        this.alarmCode = alarmCode;
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

    public String getAlarmType()
    {
        return alarmType;
    }

    public void setAlarmType(String alarmType)
    {
        this.alarmType = alarmType;
    }

    public String getAlarmLevel()
    {
        return alarmLevel;
    }

    public void setAlarmLevel(String alarmLevel)
    {
        this.alarmLevel = alarmLevel;
    }

    public String getAlarmMessage()
    {
        return alarmMessage;
    }

    public void setAlarmMessage(String alarmMessage)
    {
        this.alarmMessage = alarmMessage;
    }

    public Long getEventLogId()
    {
        return eventLogId;
    }

    public void setEventLogId(Long eventLogId)
    {
        this.eventLogId = eventLogId;
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

    public Date getAlarmTime()
    {
        return alarmTime;
    }

    public void setAlarmTime(Date alarmTime)
    {
        this.alarmTime = alarmTime;
    }

    public String getHandleStatus()
    {
        return handleStatus;
    }

    public void setHandleStatus(String handleStatus)
    {
        this.handleStatus = handleStatus;
    }

    public String getHandleBy()
    {
        return handleBy;
    }

    public void setHandleBy(String handleBy)
    {
        this.handleBy = handleBy;
    }

    public Date getHandleTime()
    {
        return handleTime;
    }

    public void setHandleTime(Date handleTime)
    {
        this.handleTime = handleTime;
    }

    public String getHandleRemark()
    {
        return handleRemark;
    }

    public void setHandleRemark(String handleRemark)
    {
        this.handleRemark = handleRemark;
    }

    @Override
    public String toString()
    {
        return new ToStringBuilder(this, ToStringStyle.MULTI_LINE_STYLE)
            .append("alarmId", getAlarmId())
            .append("alarmCode", getAlarmCode())
            .append("taskId", getTaskId())
            .append("taskCode", getTaskCode())
            .append("productCode", getProductCode())
            .append("stationCode", getStationCode())
            .append("cameraCode", getCameraCode())
            .append("alarmType", getAlarmType())
            .append("alarmLevel", getAlarmLevel())
            .append("alarmMessage", getAlarmMessage())
            .append("eventLogId", getEventLogId())
            .append("eventCode", getEventCode())
            .append("eventName", getEventName())
            .append("stepId", getStepId())
            .append("stepNo", getStepNo())
            .append("alarmTime", getAlarmTime())
            .append("handleStatus", getHandleStatus())
            .append("handleBy", getHandleBy())
            .append("handleTime", getHandleTime())
            .append("handleRemark", getHandleRemark())
            .append("createBy", getCreateBy())
            .append("createTime", getCreateTime())
            .append("updateBy", getUpdateBy())
            .append("updateTime", getUpdateTime())
            .append("remark", getRemark())
            .toString();
    }
}
