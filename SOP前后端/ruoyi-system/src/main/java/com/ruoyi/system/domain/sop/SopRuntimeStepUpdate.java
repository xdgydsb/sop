package com.ruoyi.system.domain.sop;

import java.util.Date;

public class SopRuntimeStepUpdate
{
    private Integer stepNo;
    private String stepName;
    private String expectedEvent;
    private String stepStatus;
    private String snapshotUrl;
    private String clipUrl;
    private Long clipStartMs;
    private Long clipEndMs;
    private String judgeResult;
    private String judgeMessage;
    private Date passTime;

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

    public String getStepStatus()
    {
        return stepStatus;
    }

    public void setStepStatus(String stepStatus)
    {
        this.stepStatus = stepStatus;
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

    public Date getPassTime()
    {
        return passTime;
    }

    public void setPassTime(Date passTime)
    {
        this.passTime = passTime;
    }
}
