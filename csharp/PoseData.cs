// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: pose_data.proto
#pragma warning disable 1591, 0612, 3021
#region Designer generated code

using pb = global::Google.Protobuf;
using pbc = global::Google.Protobuf.Collections;
using pbr = global::Google.Protobuf.Reflection;
using scg = global::System.Collections.Generic;
/// <summary>Holder for reflection information generated from pose_data.proto</summary>
public static partial class PoseDataReflection {

  #region Descriptor
  /// <summary>File descriptor for pose_data.proto</summary>
  public static pbr::FileDescriptor Descriptor {
    get { return descriptor; }
  }
  private static pbr::FileDescriptor descriptor;

  static PoseDataReflection() {
    byte[] descriptorData = global::System.Convert.FromBase64String(
        string.Concat(
          "Cg9wb3NlX2RhdGEucHJvdG8aG2dvb2dsZS9wcm90b2J1Zi9lbXB0eS5wcm90",
          "byIdCgVKb2ludBIJCgF4GAEgASgCEgkKAXkYAiABKAIivwMKBUh1bWFuEhQK",
          "BE5vc2UYASABKAsyBi5Kb2ludBIUCgROZWNrGAIgASgLMgYuSm9pbnQSGQoJ",
          "UlNob3VsZGVyGAMgASgLMgYuSm9pbnQSFgoGUkVsYm93GAQgASgLMgYuSm9p",
          "bnQSFgoGUldyaXN0GAUgASgLMgYuSm9pbnQSGQoJTFNob3VsZGVyGAYgASgL",
          "MgYuSm9pbnQSFgoGTEVsYm93GAcgASgLMgYuSm9pbnQSFgoGTFdyaXN0GAgg",
          "ASgLMgYuSm9pbnQSFAoEUkhpcBgJIAEoCzIGLkpvaW50EhUKBVJLbmVlGAog",
          "ASgLMgYuSm9pbnQSFgoGUkFua2xlGAsgASgLMgYuSm9pbnQSFAoETEhpcBgM",
          "IAEoCzIGLkpvaW50EhUKBUxLbmVlGA0gASgLMgYuSm9pbnQSFgoGTEFua2xl",
          "GA4gASgLMgYuSm9pbnQSFAoEUkV5ZRgPIAEoCzIGLkpvaW50EhQKBExFeWUY",
          "ECABKAsyBi5Kb2ludBIUCgRSRWFyGBEgASgLMgYuSm9pbnQSFAoETEVhchgS",
          "IAEoCzIGLkpvaW50EhIKCkNvbmZpZGVuY2UYEyABKAIiIAoGSHVtYW5zEhYK",
          "Bmh1bWFucxgBIAMoCzIGLkh1bWFuMjsKCUh1bWFuUG9zZRIuCglHZXRIdW1h",
          "bnMSFi5nb29nbGUucHJvdG9idWYuRW1wdHkaBy5IdW1hbnMiAGIGcHJvdG8z"));
    descriptor = pbr::FileDescriptor.FromGeneratedCode(descriptorData,
        new pbr::FileDescriptor[] { global::Google.Protobuf.WellKnownTypes.EmptyReflection.Descriptor, },
        new pbr::GeneratedClrTypeInfo(null, new pbr::GeneratedClrTypeInfo[] {
          new pbr::GeneratedClrTypeInfo(typeof(global::Joint), global::Joint.Parser, new[]{ "X", "Y" }, null, null, null),
          new pbr::GeneratedClrTypeInfo(typeof(global::Human), global::Human.Parser, new[]{ "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "Confidence" }, null, null, null),
          new pbr::GeneratedClrTypeInfo(typeof(global::Humans), global::Humans.Parser, new[]{ "Humans_" }, null, null, null)
        }));
  }
  #endregion

}
#region Messages
public sealed partial class Joint : pb::IMessage<Joint> {
  private static readonly pb::MessageParser<Joint> _parser = new pb::MessageParser<Joint>(() => new Joint());
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public static pb::MessageParser<Joint> Parser { get { return _parser; } }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public static pbr::MessageDescriptor Descriptor {
    get { return global::PoseDataReflection.Descriptor.MessageTypes[0]; }
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  pbr::MessageDescriptor pb::IMessage.Descriptor {
    get { return Descriptor; }
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public Joint() {
    OnConstruction();
  }

  partial void OnConstruction();

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public Joint(Joint other) : this() {
    x_ = other.x_;
    y_ = other.y_;
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public Joint Clone() {
    return new Joint(this);
  }

  /// <summary>Field number for the "x" field.</summary>
  public const int XFieldNumber = 1;
  private float x_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public float X {
    get { return x_; }
    set {
      x_ = value;
    }
  }

  /// <summary>Field number for the "y" field.</summary>
  public const int YFieldNumber = 2;
  private float y_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public float Y {
    get { return y_; }
    set {
      y_ = value;
    }
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public override bool Equals(object other) {
    return Equals(other as Joint);
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public bool Equals(Joint other) {
    if (ReferenceEquals(other, null)) {
      return false;
    }
    if (ReferenceEquals(other, this)) {
      return true;
    }
    if (X != other.X) return false;
    if (Y != other.Y) return false;
    return true;
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public override int GetHashCode() {
    int hash = 1;
    if (X != 0F) hash ^= X.GetHashCode();
    if (Y != 0F) hash ^= Y.GetHashCode();
    return hash;
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public override string ToString() {
    return pb::JsonFormatter.ToDiagnosticString(this);
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public void WriteTo(pb::CodedOutputStream output) {
    if (X != 0F) {
      output.WriteRawTag(13);
      output.WriteFloat(X);
    }
    if (Y != 0F) {
      output.WriteRawTag(21);
      output.WriteFloat(Y);
    }
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public int CalculateSize() {
    int size = 0;
    if (X != 0F) {
      size += 1 + 4;
    }
    if (Y != 0F) {
      size += 1 + 4;
    }
    return size;
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public void MergeFrom(Joint other) {
    if (other == null) {
      return;
    }
    if (other.X != 0F) {
      X = other.X;
    }
    if (other.Y != 0F) {
      Y = other.Y;
    }
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public void MergeFrom(pb::CodedInputStream input) {
    uint tag;
    while ((tag = input.ReadTag()) != 0) {
      switch(tag) {
        default:
          input.SkipLastField();
          break;
        case 13: {
          X = input.ReadFloat();
          break;
        }
        case 21: {
          Y = input.ReadFloat();
          break;
        }
      }
    }
  }

}

public sealed partial class Human : pb::IMessage<Human> {
  private static readonly pb::MessageParser<Human> _parser = new pb::MessageParser<Human>(() => new Human());
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public static pb::MessageParser<Human> Parser { get { return _parser; } }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public static pbr::MessageDescriptor Descriptor {
    get { return global::PoseDataReflection.Descriptor.MessageTypes[1]; }
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  pbr::MessageDescriptor pb::IMessage.Descriptor {
    get { return Descriptor; }
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public Human() {
    OnConstruction();
  }

  partial void OnConstruction();

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public Human(Human other) : this() {
    Nose = other.nose_ != null ? other.Nose.Clone() : null;
    Neck = other.neck_ != null ? other.Neck.Clone() : null;
    RShoulder = other.rShoulder_ != null ? other.RShoulder.Clone() : null;
    RElbow = other.rElbow_ != null ? other.RElbow.Clone() : null;
    RWrist = other.rWrist_ != null ? other.RWrist.Clone() : null;
    LShoulder = other.lShoulder_ != null ? other.LShoulder.Clone() : null;
    LElbow = other.lElbow_ != null ? other.LElbow.Clone() : null;
    LWrist = other.lWrist_ != null ? other.LWrist.Clone() : null;
    RHip = other.rHip_ != null ? other.RHip.Clone() : null;
    RKnee = other.rKnee_ != null ? other.RKnee.Clone() : null;
    RAnkle = other.rAnkle_ != null ? other.RAnkle.Clone() : null;
    LHip = other.lHip_ != null ? other.LHip.Clone() : null;
    LKnee = other.lKnee_ != null ? other.LKnee.Clone() : null;
    LAnkle = other.lAnkle_ != null ? other.LAnkle.Clone() : null;
    REye = other.rEye_ != null ? other.REye.Clone() : null;
    LEye = other.lEye_ != null ? other.LEye.Clone() : null;
    REar = other.rEar_ != null ? other.REar.Clone() : null;
    LEar = other.lEar_ != null ? other.LEar.Clone() : null;
    confidence_ = other.confidence_;
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public Human Clone() {
    return new Human(this);
  }

  /// <summary>Field number for the "Nose" field.</summary>
  public const int NoseFieldNumber = 1;
  private global::Joint nose_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint Nose {
    get { return nose_; }
    set {
      nose_ = value;
    }
  }

  /// <summary>Field number for the "Neck" field.</summary>
  public const int NeckFieldNumber = 2;
  private global::Joint neck_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint Neck {
    get { return neck_; }
    set {
      neck_ = value;
    }
  }

  /// <summary>Field number for the "RShoulder" field.</summary>
  public const int RShoulderFieldNumber = 3;
  private global::Joint rShoulder_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint RShoulder {
    get { return rShoulder_; }
    set {
      rShoulder_ = value;
    }
  }

  /// <summary>Field number for the "RElbow" field.</summary>
  public const int RElbowFieldNumber = 4;
  private global::Joint rElbow_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint RElbow {
    get { return rElbow_; }
    set {
      rElbow_ = value;
    }
  }

  /// <summary>Field number for the "RWrist" field.</summary>
  public const int RWristFieldNumber = 5;
  private global::Joint rWrist_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint RWrist {
    get { return rWrist_; }
    set {
      rWrist_ = value;
    }
  }

  /// <summary>Field number for the "LShoulder" field.</summary>
  public const int LShoulderFieldNumber = 6;
  private global::Joint lShoulder_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint LShoulder {
    get { return lShoulder_; }
    set {
      lShoulder_ = value;
    }
  }

  /// <summary>Field number for the "LElbow" field.</summary>
  public const int LElbowFieldNumber = 7;
  private global::Joint lElbow_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint LElbow {
    get { return lElbow_; }
    set {
      lElbow_ = value;
    }
  }

  /// <summary>Field number for the "LWrist" field.</summary>
  public const int LWristFieldNumber = 8;
  private global::Joint lWrist_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint LWrist {
    get { return lWrist_; }
    set {
      lWrist_ = value;
    }
  }

  /// <summary>Field number for the "RHip" field.</summary>
  public const int RHipFieldNumber = 9;
  private global::Joint rHip_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint RHip {
    get { return rHip_; }
    set {
      rHip_ = value;
    }
  }

  /// <summary>Field number for the "RKnee" field.</summary>
  public const int RKneeFieldNumber = 10;
  private global::Joint rKnee_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint RKnee {
    get { return rKnee_; }
    set {
      rKnee_ = value;
    }
  }

  /// <summary>Field number for the "RAnkle" field.</summary>
  public const int RAnkleFieldNumber = 11;
  private global::Joint rAnkle_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint RAnkle {
    get { return rAnkle_; }
    set {
      rAnkle_ = value;
    }
  }

  /// <summary>Field number for the "LHip" field.</summary>
  public const int LHipFieldNumber = 12;
  private global::Joint lHip_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint LHip {
    get { return lHip_; }
    set {
      lHip_ = value;
    }
  }

  /// <summary>Field number for the "LKnee" field.</summary>
  public const int LKneeFieldNumber = 13;
  private global::Joint lKnee_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint LKnee {
    get { return lKnee_; }
    set {
      lKnee_ = value;
    }
  }

  /// <summary>Field number for the "LAnkle" field.</summary>
  public const int LAnkleFieldNumber = 14;
  private global::Joint lAnkle_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint LAnkle {
    get { return lAnkle_; }
    set {
      lAnkle_ = value;
    }
  }

  /// <summary>Field number for the "REye" field.</summary>
  public const int REyeFieldNumber = 15;
  private global::Joint rEye_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint REye {
    get { return rEye_; }
    set {
      rEye_ = value;
    }
  }

  /// <summary>Field number for the "LEye" field.</summary>
  public const int LEyeFieldNumber = 16;
  private global::Joint lEye_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint LEye {
    get { return lEye_; }
    set {
      lEye_ = value;
    }
  }

  /// <summary>Field number for the "REar" field.</summary>
  public const int REarFieldNumber = 17;
  private global::Joint rEar_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint REar {
    get { return rEar_; }
    set {
      rEar_ = value;
    }
  }

  /// <summary>Field number for the "LEar" field.</summary>
  public const int LEarFieldNumber = 18;
  private global::Joint lEar_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::Joint LEar {
    get { return lEar_; }
    set {
      lEar_ = value;
    }
  }

  /// <summary>Field number for the "Confidence" field.</summary>
  public const int ConfidenceFieldNumber = 19;
  private float confidence_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public float Confidence {
    get { return confidence_; }
    set {
      confidence_ = value;
    }
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public override bool Equals(object other) {
    return Equals(other as Human);
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public bool Equals(Human other) {
    if (ReferenceEquals(other, null)) {
      return false;
    }
    if (ReferenceEquals(other, this)) {
      return true;
    }
    if (!object.Equals(Nose, other.Nose)) return false;
    if (!object.Equals(Neck, other.Neck)) return false;
    if (!object.Equals(RShoulder, other.RShoulder)) return false;
    if (!object.Equals(RElbow, other.RElbow)) return false;
    if (!object.Equals(RWrist, other.RWrist)) return false;
    if (!object.Equals(LShoulder, other.LShoulder)) return false;
    if (!object.Equals(LElbow, other.LElbow)) return false;
    if (!object.Equals(LWrist, other.LWrist)) return false;
    if (!object.Equals(RHip, other.RHip)) return false;
    if (!object.Equals(RKnee, other.RKnee)) return false;
    if (!object.Equals(RAnkle, other.RAnkle)) return false;
    if (!object.Equals(LHip, other.LHip)) return false;
    if (!object.Equals(LKnee, other.LKnee)) return false;
    if (!object.Equals(LAnkle, other.LAnkle)) return false;
    if (!object.Equals(REye, other.REye)) return false;
    if (!object.Equals(LEye, other.LEye)) return false;
    if (!object.Equals(REar, other.REar)) return false;
    if (!object.Equals(LEar, other.LEar)) return false;
    if (Confidence != other.Confidence) return false;
    return true;
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public override int GetHashCode() {
    int hash = 1;
    if (nose_ != null) hash ^= Nose.GetHashCode();
    if (neck_ != null) hash ^= Neck.GetHashCode();
    if (rShoulder_ != null) hash ^= RShoulder.GetHashCode();
    if (rElbow_ != null) hash ^= RElbow.GetHashCode();
    if (rWrist_ != null) hash ^= RWrist.GetHashCode();
    if (lShoulder_ != null) hash ^= LShoulder.GetHashCode();
    if (lElbow_ != null) hash ^= LElbow.GetHashCode();
    if (lWrist_ != null) hash ^= LWrist.GetHashCode();
    if (rHip_ != null) hash ^= RHip.GetHashCode();
    if (rKnee_ != null) hash ^= RKnee.GetHashCode();
    if (rAnkle_ != null) hash ^= RAnkle.GetHashCode();
    if (lHip_ != null) hash ^= LHip.GetHashCode();
    if (lKnee_ != null) hash ^= LKnee.GetHashCode();
    if (lAnkle_ != null) hash ^= LAnkle.GetHashCode();
    if (rEye_ != null) hash ^= REye.GetHashCode();
    if (lEye_ != null) hash ^= LEye.GetHashCode();
    if (rEar_ != null) hash ^= REar.GetHashCode();
    if (lEar_ != null) hash ^= LEar.GetHashCode();
    if (Confidence != 0F) hash ^= Confidence.GetHashCode();
    return hash;
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public override string ToString() {
    return pb::JsonFormatter.ToDiagnosticString(this);
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public void WriteTo(pb::CodedOutputStream output) {
    if (nose_ != null) {
      output.WriteRawTag(10);
      output.WriteMessage(Nose);
    }
    if (neck_ != null) {
      output.WriteRawTag(18);
      output.WriteMessage(Neck);
    }
    if (rShoulder_ != null) {
      output.WriteRawTag(26);
      output.WriteMessage(RShoulder);
    }
    if (rElbow_ != null) {
      output.WriteRawTag(34);
      output.WriteMessage(RElbow);
    }
    if (rWrist_ != null) {
      output.WriteRawTag(42);
      output.WriteMessage(RWrist);
    }
    if (lShoulder_ != null) {
      output.WriteRawTag(50);
      output.WriteMessage(LShoulder);
    }
    if (lElbow_ != null) {
      output.WriteRawTag(58);
      output.WriteMessage(LElbow);
    }
    if (lWrist_ != null) {
      output.WriteRawTag(66);
      output.WriteMessage(LWrist);
    }
    if (rHip_ != null) {
      output.WriteRawTag(74);
      output.WriteMessage(RHip);
    }
    if (rKnee_ != null) {
      output.WriteRawTag(82);
      output.WriteMessage(RKnee);
    }
    if (rAnkle_ != null) {
      output.WriteRawTag(90);
      output.WriteMessage(RAnkle);
    }
    if (lHip_ != null) {
      output.WriteRawTag(98);
      output.WriteMessage(LHip);
    }
    if (lKnee_ != null) {
      output.WriteRawTag(106);
      output.WriteMessage(LKnee);
    }
    if (lAnkle_ != null) {
      output.WriteRawTag(114);
      output.WriteMessage(LAnkle);
    }
    if (rEye_ != null) {
      output.WriteRawTag(122);
      output.WriteMessage(REye);
    }
    if (lEye_ != null) {
      output.WriteRawTag(130, 1);
      output.WriteMessage(LEye);
    }
    if (rEar_ != null) {
      output.WriteRawTag(138, 1);
      output.WriteMessage(REar);
    }
    if (lEar_ != null) {
      output.WriteRawTag(146, 1);
      output.WriteMessage(LEar);
    }
    if (Confidence != 0F) {
      output.WriteRawTag(157, 1);
      output.WriteFloat(Confidence);
    }
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public int CalculateSize() {
    int size = 0;
    if (nose_ != null) {
      size += 1 + pb::CodedOutputStream.ComputeMessageSize(Nose);
    }
    if (neck_ != null) {
      size += 1 + pb::CodedOutputStream.ComputeMessageSize(Neck);
    }
    if (rShoulder_ != null) {
      size += 1 + pb::CodedOutputStream.ComputeMessageSize(RShoulder);
    }
    if (rElbow_ != null) {
      size += 1 + pb::CodedOutputStream.ComputeMessageSize(RElbow);
    }
    if (rWrist_ != null) {
      size += 1 + pb::CodedOutputStream.ComputeMessageSize(RWrist);
    }
    if (lShoulder_ != null) {
      size += 1 + pb::CodedOutputStream.ComputeMessageSize(LShoulder);
    }
    if (lElbow_ != null) {
      size += 1 + pb::CodedOutputStream.ComputeMessageSize(LElbow);
    }
    if (lWrist_ != null) {
      size += 1 + pb::CodedOutputStream.ComputeMessageSize(LWrist);
    }
    if (rHip_ != null) {
      size += 1 + pb::CodedOutputStream.ComputeMessageSize(RHip);
    }
    if (rKnee_ != null) {
      size += 1 + pb::CodedOutputStream.ComputeMessageSize(RKnee);
    }
    if (rAnkle_ != null) {
      size += 1 + pb::CodedOutputStream.ComputeMessageSize(RAnkle);
    }
    if (lHip_ != null) {
      size += 1 + pb::CodedOutputStream.ComputeMessageSize(LHip);
    }
    if (lKnee_ != null) {
      size += 1 + pb::CodedOutputStream.ComputeMessageSize(LKnee);
    }
    if (lAnkle_ != null) {
      size += 1 + pb::CodedOutputStream.ComputeMessageSize(LAnkle);
    }
    if (rEye_ != null) {
      size += 1 + pb::CodedOutputStream.ComputeMessageSize(REye);
    }
    if (lEye_ != null) {
      size += 2 + pb::CodedOutputStream.ComputeMessageSize(LEye);
    }
    if (rEar_ != null) {
      size += 2 + pb::CodedOutputStream.ComputeMessageSize(REar);
    }
    if (lEar_ != null) {
      size += 2 + pb::CodedOutputStream.ComputeMessageSize(LEar);
    }
    if (Confidence != 0F) {
      size += 2 + 4;
    }
    return size;
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public void MergeFrom(Human other) {
    if (other == null) {
      return;
    }
    if (other.nose_ != null) {
      if (nose_ == null) {
        nose_ = new global::Joint();
      }
      Nose.MergeFrom(other.Nose);
    }
    if (other.neck_ != null) {
      if (neck_ == null) {
        neck_ = new global::Joint();
      }
      Neck.MergeFrom(other.Neck);
    }
    if (other.rShoulder_ != null) {
      if (rShoulder_ == null) {
        rShoulder_ = new global::Joint();
      }
      RShoulder.MergeFrom(other.RShoulder);
    }
    if (other.rElbow_ != null) {
      if (rElbow_ == null) {
        rElbow_ = new global::Joint();
      }
      RElbow.MergeFrom(other.RElbow);
    }
    if (other.rWrist_ != null) {
      if (rWrist_ == null) {
        rWrist_ = new global::Joint();
      }
      RWrist.MergeFrom(other.RWrist);
    }
    if (other.lShoulder_ != null) {
      if (lShoulder_ == null) {
        lShoulder_ = new global::Joint();
      }
      LShoulder.MergeFrom(other.LShoulder);
    }
    if (other.lElbow_ != null) {
      if (lElbow_ == null) {
        lElbow_ = new global::Joint();
      }
      LElbow.MergeFrom(other.LElbow);
    }
    if (other.lWrist_ != null) {
      if (lWrist_ == null) {
        lWrist_ = new global::Joint();
      }
      LWrist.MergeFrom(other.LWrist);
    }
    if (other.rHip_ != null) {
      if (rHip_ == null) {
        rHip_ = new global::Joint();
      }
      RHip.MergeFrom(other.RHip);
    }
    if (other.rKnee_ != null) {
      if (rKnee_ == null) {
        rKnee_ = new global::Joint();
      }
      RKnee.MergeFrom(other.RKnee);
    }
    if (other.rAnkle_ != null) {
      if (rAnkle_ == null) {
        rAnkle_ = new global::Joint();
      }
      RAnkle.MergeFrom(other.RAnkle);
    }
    if (other.lHip_ != null) {
      if (lHip_ == null) {
        lHip_ = new global::Joint();
      }
      LHip.MergeFrom(other.LHip);
    }
    if (other.lKnee_ != null) {
      if (lKnee_ == null) {
        lKnee_ = new global::Joint();
      }
      LKnee.MergeFrom(other.LKnee);
    }
    if (other.lAnkle_ != null) {
      if (lAnkle_ == null) {
        lAnkle_ = new global::Joint();
      }
      LAnkle.MergeFrom(other.LAnkle);
    }
    if (other.rEye_ != null) {
      if (rEye_ == null) {
        rEye_ = new global::Joint();
      }
      REye.MergeFrom(other.REye);
    }
    if (other.lEye_ != null) {
      if (lEye_ == null) {
        lEye_ = new global::Joint();
      }
      LEye.MergeFrom(other.LEye);
    }
    if (other.rEar_ != null) {
      if (rEar_ == null) {
        rEar_ = new global::Joint();
      }
      REar.MergeFrom(other.REar);
    }
    if (other.lEar_ != null) {
      if (lEar_ == null) {
        lEar_ = new global::Joint();
      }
      LEar.MergeFrom(other.LEar);
    }
    if (other.Confidence != 0F) {
      Confidence = other.Confidence;
    }
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public void MergeFrom(pb::CodedInputStream input) {
    uint tag;
    while ((tag = input.ReadTag()) != 0) {
      switch(tag) {
        default:
          input.SkipLastField();
          break;
        case 10: {
          if (nose_ == null) {
            nose_ = new global::Joint();
          }
          input.ReadMessage(nose_);
          break;
        }
        case 18: {
          if (neck_ == null) {
            neck_ = new global::Joint();
          }
          input.ReadMessage(neck_);
          break;
        }
        case 26: {
          if (rShoulder_ == null) {
            rShoulder_ = new global::Joint();
          }
          input.ReadMessage(rShoulder_);
          break;
        }
        case 34: {
          if (rElbow_ == null) {
            rElbow_ = new global::Joint();
          }
          input.ReadMessage(rElbow_);
          break;
        }
        case 42: {
          if (rWrist_ == null) {
            rWrist_ = new global::Joint();
          }
          input.ReadMessage(rWrist_);
          break;
        }
        case 50: {
          if (lShoulder_ == null) {
            lShoulder_ = new global::Joint();
          }
          input.ReadMessage(lShoulder_);
          break;
        }
        case 58: {
          if (lElbow_ == null) {
            lElbow_ = new global::Joint();
          }
          input.ReadMessage(lElbow_);
          break;
        }
        case 66: {
          if (lWrist_ == null) {
            lWrist_ = new global::Joint();
          }
          input.ReadMessage(lWrist_);
          break;
        }
        case 74: {
          if (rHip_ == null) {
            rHip_ = new global::Joint();
          }
          input.ReadMessage(rHip_);
          break;
        }
        case 82: {
          if (rKnee_ == null) {
            rKnee_ = new global::Joint();
          }
          input.ReadMessage(rKnee_);
          break;
        }
        case 90: {
          if (rAnkle_ == null) {
            rAnkle_ = new global::Joint();
          }
          input.ReadMessage(rAnkle_);
          break;
        }
        case 98: {
          if (lHip_ == null) {
            lHip_ = new global::Joint();
          }
          input.ReadMessage(lHip_);
          break;
        }
        case 106: {
          if (lKnee_ == null) {
            lKnee_ = new global::Joint();
          }
          input.ReadMessage(lKnee_);
          break;
        }
        case 114: {
          if (lAnkle_ == null) {
            lAnkle_ = new global::Joint();
          }
          input.ReadMessage(lAnkle_);
          break;
        }
        case 122: {
          if (rEye_ == null) {
            rEye_ = new global::Joint();
          }
          input.ReadMessage(rEye_);
          break;
        }
        case 130: {
          if (lEye_ == null) {
            lEye_ = new global::Joint();
          }
          input.ReadMessage(lEye_);
          break;
        }
        case 138: {
          if (rEar_ == null) {
            rEar_ = new global::Joint();
          }
          input.ReadMessage(rEar_);
          break;
        }
        case 146: {
          if (lEar_ == null) {
            lEar_ = new global::Joint();
          }
          input.ReadMessage(lEar_);
          break;
        }
        case 157: {
          Confidence = input.ReadFloat();
          break;
        }
      }
    }
  }

}

public sealed partial class Humans : pb::IMessage<Humans> {
  private static readonly pb::MessageParser<Humans> _parser = new pb::MessageParser<Humans>(() => new Humans());
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public static pb::MessageParser<Humans> Parser { get { return _parser; } }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public static pbr::MessageDescriptor Descriptor {
    get { return global::PoseDataReflection.Descriptor.MessageTypes[2]; }
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  pbr::MessageDescriptor pb::IMessage.Descriptor {
    get { return Descriptor; }
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public Humans() {
    OnConstruction();
  }

  partial void OnConstruction();

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public Humans(Humans other) : this() {
    humans_ = other.humans_.Clone();
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public Humans Clone() {
    return new Humans(this);
  }

  /// <summary>Field number for the "humans" field.</summary>
  public const int Humans_FieldNumber = 1;
  private static readonly pb::FieldCodec<global::Human> _repeated_humans_codec
      = pb::FieldCodec.ForMessage(10, global::Human.Parser);
  private readonly pbc::RepeatedField<global::Human> humans_ = new pbc::RepeatedField<global::Human>();
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public pbc::RepeatedField<global::Human> Humans_ {
    get { return humans_; }
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public override bool Equals(object other) {
    return Equals(other as Humans);
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public bool Equals(Humans other) {
    if (ReferenceEquals(other, null)) {
      return false;
    }
    if (ReferenceEquals(other, this)) {
      return true;
    }
    if(!humans_.Equals(other.humans_)) return false;
    return true;
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public override int GetHashCode() {
    int hash = 1;
    hash ^= humans_.GetHashCode();
    return hash;
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public override string ToString() {
    return pb::JsonFormatter.ToDiagnosticString(this);
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public void WriteTo(pb::CodedOutputStream output) {
    humans_.WriteTo(output, _repeated_humans_codec);
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public int CalculateSize() {
    int size = 0;
    size += humans_.CalculateSize(_repeated_humans_codec);
    return size;
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public void MergeFrom(Humans other) {
    if (other == null) {
      return;
    }
    humans_.Add(other.humans_);
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public void MergeFrom(pb::CodedInputStream input) {
    uint tag;
    while ((tag = input.ReadTag()) != 0) {
      switch(tag) {
        default:
          input.SkipLastField();
          break;
        case 10: {
          humans_.AddEntriesFrom(input, _repeated_humans_codec);
          break;
        }
      }
    }
  }

}

#endregion


#endregion Designer generated code