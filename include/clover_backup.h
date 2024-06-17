
#pragma once

#include <algorithm>
#include <clover_field.h>

namespace quda
{

  struct CloverBundleBackup {
    CloverField *precise = nullptr;
    CloverField *sloppy = nullptr;
    CloverField *precondition = nullptr;
    CloverField *refinement = nullptr;
    CloverField *eigensolver = nullptr;

    CloverBundleBackup() = default;

    void backup(CloverField *precise_, CloverField *sloppy_, CloverField *precondition_, CloverField *refinement_,
                CloverField *eigensolver_)
    {
      if (precise_) precise = new CloverField(*precise_); // Copy it

      // sloppy_ can be just pointing to precise_ so mirror that here.
      if (sloppy_ == precise_)
        sloppy = precise;
      else {
        sloppy = new CloverField(*sloppy_);
      }

      // precondition can be either precise or sloppy or its own things
      if (precondition_ == precise_)
        precondition = precise;
      else if (precondition_ == sloppy_)
        precondition = sloppy;
      else {
        precondition = new CloverField(*precondition_);
      }

      // refinement  can be possibkly sloppy
      if (refinement_ == sloppy_)
        refinement = sloppy;
      else {
        refinement = new CloverField(*refinement_);
      }

      // eigensolver can be precise, precondtion, or sloppy
      if (eigensolver_ == precise_)
        eigensolver = precise;
      else if (eigensolver_ == precondition_)
        eigensolver = precondition;
      else if (eigensolver_ == sloppy_)
        eigensolver = sloppy;
      else {
        eigensolver = new CloverField(*eigensolver_);
      }
    }
  }; // Class

  void setupCloverFields(CloverField *collected_clover, CloverField *&precise, CloverField *&sloppy,
                         CloverField *&precondition, CloverField *&refinement, CloverField *&eigensolver,
                         const CloverBundleBackup &bkup)
  {
    // First things first. The new collected gauge is going to become the 'precise'
    // Things to check: what to do about precise first (We copied it so we can free it)
    // Do I need to copy collected?
    if (precise) delete (precise);
    precise = collected_clover;

    CloverFieldParam precise_param(*collected_clover);

    // Now Sloppy can alias precise -- we can check this from the bkup
    if (bkup.sloppy == bkup.precise) {
      // Sloppy was pointing to precise so
      // so we can do the same
      sloppy = precise;
    } else {
      // Sloppy was its own thing -- let's free it
      if (sloppy) delete sloppy;

      // We can get its precision etcs all from the old sloppy parameters
      CloverFieldParam sloppy_param(*(bkup.sloppy));
      sloppy_param.create = QUDA_NULL_FIELD_CREATE;
      // We need to resize and pad
      sloppy_param.x = precise_param.x;
      sloppy_param.pad = precise_param.pad;

      // we need to resize this based on precise (which is the collected gauge)
      sloppy = new CloverField(sloppy_param);
      sloppy->copy(*precise); // This copy should trim the precisions etc
    }

    if (bkup.precondition == bkup.precise) {
      precondition = precise;
    } else if (bkup.precondition == bkup.sloppy) {
      precondition = sloppy;
    } else {
      if (precondition) delete precondition;

      CloverFieldParam precondition_param(*(bkup.precondition));
      precondition_param.create = QUDA_NULL_FIELD_CREATE;
      precondition_param.x = precise_param.x;
      precondition_param.pad = precise_param.pad;
      precondition = new CloverField(precondition_param);
      precondition->copy(*precise);
    }

    // refinement  can be possibkly sloppy
    if (bkup.refinement == bkup.sloppy)
      refinement = sloppy;
    else {
      if (refinement) delete refinement;
      CloverFieldParam refinement_param(*(bkup.refinement));
      refinement_param.create = QUDA_NULL_FIELD_CREATE;
      refinement_param.x = precise_param.x;
      refinement_param.pad = precise_param.pad;
      refinement = new CloverField(refinement_param);
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
      CloverFieldParam eigensolver_param(*(bkup.eigensolver));
      eigensolver_param.create = QUDA_NULL_FIELD_CREATE;
      eigensolver_param.x = precise_param.x;
      eigensolver_param.pad = precise_param.pad;

      eigensolver = new CloverField(eigensolver_param);
      eigensolver->copy(*precise);
    }
  }
} // namespace quda
