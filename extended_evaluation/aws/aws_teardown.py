"""Delete the Lambda function and its CloudWatch log group (restricted profile; only dowlab-* names)."""
import boto3
s = boto3.Session(profile_name="dow-lab", region_name="us-east-1")
for name, fn in (("function", lambda: s.client("lambda").delete_function(FunctionName="dowlab-work")),
                 ("log group", lambda: s.client("logs").delete_log_group(logGroupName="/aws/lambda/dowlab-work"))):
    try:
        fn(); print("deleted", name)
    except Exception as e:
        print(name, "->", type(e).__name__, str(e)[:100])
