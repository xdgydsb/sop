package com.ruoyi.system.domain.sop;

import java.math.BigDecimal;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;
import com.ruoyi.common.annotation.Excel;
import com.ruoyi.common.core.domain.BaseEntity;

/**
 * SOP step object sop_step
 *
 * @author ruoyi
 */
public class SopStep extends BaseEntity
{
    private static final long serialVersionUID = 1L;

    /** Step ID */
    private Long stepId;

    /** SOP ID */
    @Excel(name = "SOP ID")
    private Long sopId;

    /** SOP name */
    @Excel(name = "SOP名称")
    private String sopName;

    /** Step sequence number */
    @Excel(name = "步骤序号")
    private Integer stepNo;

    /** Step name */
    @Excel(name = "步骤名称")
    private String stepName;

    /** Expected event code */
    @Excel(name = "期望事件编码")
    private String expectedEvent;

    /** Required confidence */
    @Excel(name = "最低置信度")
    private BigDecimal requiredConfidence;

    /** Standard duration in seconds */
    @Excel(name = "标准时长")
    private Integer standardDuration;

    /** Status: 0 normal, 1 disabled */
    @Excel(name = "状态", readConverterExp = "0=正常,1=停用")
    private String status;

    public Long getStepId()
    {
        return stepId;
    }

    public void setStepId(Long stepId)
    {
        this.stepId = stepId;
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

    public Integer getStandardDuration()
    {
        return standardDuration;
    }

    public void setStandardDuration(Integer standardDuration)
    {
        this.standardDuration = standardDuration;
    }

    public String getStatus()
    {
        return status;
    }

    public void setStatus(String status)
    {
        this.status = status;
    }

    @Override
    public String toString()
    {
        return new ToStringBuilder(this, ToStringStyle.MULTI_LINE_STYLE)
            .append("stepId", getStepId())
            .append("sopId", getSopId())
            .append("sopName", getSopName())
            .append("stepNo", getStepNo())
            .append("stepName", getStepName())
            .append("expectedEvent", getExpectedEvent())
            .append("requiredConfidence", getRequiredConfidence())
            .append("standardDuration", getStandardDuration())
            .append("status", getStatus())
            .append("createBy", getCreateBy())
            .append("createTime", getCreateTime())
            .append("updateBy", getUpdateBy())
            .append("updateTime", getUpdateTime())
            .append("remark", getRemark())
            .toString();
    }
}
