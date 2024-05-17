#pragma once

#include <algorithm>
#include <gauge_field.h>

namespace quda
{

  struct GaugeBundleBackup {
    GaugeField *precise;
    GaugeField *sloppy;
    GaugeField *precondition;
    GaugeField *refinement;
    GaugeField *eigensolver;
    GaugeField *extended;

    GaugeBundleBackup() :
      precise(nullptr), sloppy(nullptr), precondition(nullptr), refinement(nullptr), eigensolver(nullptr), extended(nullptr)
    {
    }

    void backup(GaugeField *precise_, GaugeField *sloppy_, GaugeField *precondition_, GaugeField *refinement_,
                GaugeField *eigensolver_, GaugeField *extended_)
    {
      if (precise_) precise = new GaugeField(*precise_); // Copy it

      // sloppy_ can be just pointing to precise_ so mirror that here.
      if (sloppy_ == precise_)
        sloppy = precise;
      else {
        sloppy = new GaugeField(*sloppy_);
      }

      // precondition can be either precise or sloppy or its own things
      if (precondition_ == precise_)
        precondition = precise;
      else if (precondition_ == sloppy_)
        precondition = sloppy;
      else {
        precondition = new GaugeField(*precondition_);
      }

      // refinement  can be possibkly sloppy
      if (refinement_ == sloppy_)
        refinement = sloppy;
      else {
        refinement = new GaugeField(*refinement_);
      }

      // eigensolver can be precise, precondtion, or sloppy
      if (eigensolver_ == precise_)
        eigensolver = precise;
      else if (eigensolver_ == precondition_)
        eigensolver = precondition;
      else if (eigensolver_ == sloppy_)
        eigensolver = sloppy;
      else {
        eigensolver = new GaugeField(*eigensolver_);
      }

      if (extended_) extended = new GaugeField(*extended_);
    }

#if 0
    void restore(GaugeField **precise_, GaugeField **sloppy_, GaugeField **precondition_, GaugeField **refinement_,
                 GaugeField **eigensolver_, GaugeField **extended_)
    {

      // If I have a precise
      if (precise) {
        GaugeField *precise_in = *precise_;
        std::swap(precise, precise_in); // Swap
        *precise_ = precise_in;
        if (precise) delete precise; // if not null delete input one
      }

      // Sloppy can alias precise, which has been swapped into *precise_
      if (sloppy == *precise_)
        *sloppy_ = *precise_;
      else {
        GaugeField *sloppy_in = *sloppy_;
        std::swap(sloppy, sloppy_in);
        *sloppy_ = sloppy_in;
        if (sloppy) delete sloppy;
      }

      // precondition can alias  either precise or sloppy which are now *xxx_
      if (precondition == *precise_)
        *precondition_ = *precise_;
      else if (precondition == *sloppy_)
        *precondition_ = *sloppy_;
      else {
        GaugeField *precondition_in = *precondition_;
        std::swap(precondition, precondition_in);
        *precondition_ = precondition_in;

        if (precondition) delete precondition;
      }

      if (refinement == *sloppy_) {
        *refinement_ = *sloppy_;
      } else {
        GaugeField *refinement_in = *refinement_;
        std::swap(refinement, refinement_in);
        *refinement_ = refinement_in;
        if (refinement) delete refinement;
      }

      // eigensolver can be precise, precondtion, or sloppy
      if (eigensolver == *precise_)
        *eigensolver_ = *precise_;
      else if (eigensolver == *precondition_)
        *eigensolver_ = *precondition_;
      else if (eigensolver == *sloppy_)
        *eigensolver_ = *sloppy_;
      else {
        GaugeField *eigensolver_in = *eigensolver_;
        std::swap(eigensolver, eigensolver_in);
        *eigensolver_ = eigensolver_in;
        if (eigensolver) delete eigensolver;
      }

      if (extended) {
        GaugeField *extended_in = *extended_;
        std::swap(extended, extended_in);
        *extended_ = extended_in ;
        if (extended) delete extended;
      }
    }
#endif

  }; // Class

  void setupGaugeFields(GaugeField *collected_gauge, GaugeField *&precise, GaugeField *&sloppy,
                        GaugeField *&precondition, GaugeField *&refinement, GaugeField *&eigensolver,
                        GaugeField *&extended, const GaugeBundleBackup &bkup, TimeProfile &profile)
  {
    // First things first. The new collected gauge is going to become the 'precise'
    // Things to check: what to do about precise first (We copied it so we can free it)
    // Do I need to copy collected?
    if (precise) delete (precise);
    precise = collected_gauge;
    precise->exchangeGhost();

    GaugeFieldParam precise_param(*collected_gauge);

    // Now Sloppy can alias precise -- we can check this from the bkup
    if (bkup.sloppy == bkup.precise) {
      // Sloppy was pointing to precise so
      // so we can do the same
      sloppy = precise;
    } else {
      // Sloppy was its own thing -- let's free it
      if (sloppy) delete sloppy;

      // We can get its precision etcs all from the old sloppy parameters
      GaugeFieldParam sloppy_param(*(bkup.sloppy));
      sloppy_param.create = QUDA_NULL_FIELD_CREATE;
      // We need to resize and pad
      sloppy_param.x = precise_param.x;
      sloppy_param.pad = precise_param.pad;

      // we need to resize this based on precise (which is the collected gauge)
      sloppy = new GaugeField(sloppy_param);
      sloppy->copy(*precise); // This copy should trim the precisions etc
    }

    if (bkup.precondition == bkup.precise) {
      precondition = precise;
    } else if (bkup.precondition == bkup.sloppy) {
      precondition = sloppy;
    } else {
      if (precondition) delete precondition;

      GaugeFieldParam precondition_param(*(bkup.precondition));
      precondition_param.create = QUDA_NULL_FIELD_CREATE;
      precondition_param.x = precise_param.x;
      precondition_param.pad = precise_param.pad;
      precondition = new GaugeField(precondition_param);
      precondition->copy(*precise);
    }

    // refinement  can be possibkly sloppy
    if (bkup.refinement == bkup.sloppy)
      refinement = sloppy;
    else {
      if (refinement) delete refinement;
      GaugeFieldParam refinement_param(*(bkup.refinement));
      refinement_param.create = QUDA_NULL_FIELD_CREATE;
      refinement_param.x = precise_param.x;
      refinement_param.pad = precise_param.pad;
      refinement = new GaugeField(refinement_param);
      refinement->copy(*precise);
    }

    // eigensolver can be precise, precondtion, or sloppy
    if (bkup.eigensolver == bkup.precise)
      eigensolver = precise;
    else if (bkup.eigensolver == bkup.precondition)
      eigensolver = precondition;
    else if (bkup.eigensolver == bkup.sloppy)
      eigensolver = sloppy;
    else {
      if (eigensolver) delete eigensolver;
      GaugeFieldParam eigensolver_param(*(bkup.eigensolver));
      eigensolver_param.create = QUDA_NULL_FIELD_CREATE;
      eigensolver_param.x = precise_param.x;
      eigensolver_param.pad = precise_param.pad;

      eigensolver = new GaugeField(eigensolver_param);
      eigensolver->copy(*precise);
    }

    if (bkup.extended) {

      if (extended) delete extended;
      // I need this to grab the R
      GaugeFieldParam extended_param(*(bkup.extended));
      auto R = extended_param.r;

      // Now I can just call createExtendedGauge -- precondition is already resized
      extended = createExtendedGauge(*precondition, R, profile);
    }
  }
} // namespace quda
